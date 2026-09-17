import polars as pl
import numpy as np
import pandas as pd
import duckdb


# Load devices back from disk
device_id_train = pd.read_csv('train/device_id_train.csv')
device_id_valid = pd.read_csv('train/device_id_valid.csv')


# Process target data
target = pd.read_csv('train/train_config.csv')

# Create one-hot encoded columns for Phase
phase_dummies = pd.get_dummies(target['Phase'], prefix='Phase')

# Create one-hot encoded columns for Function
function_dummies = pd.get_dummies(target['Function'])

# Combine the results with original identifier columns
target = pd.concat([
    target[['DeviceId', 'Detector']],  # Keep identifier columns
    phase_dummies,                      # Add Phase columns
    function_dummies                    # Add Function columns
], axis=1)

# Convert Detector to int8 for memory efficiency
target['Detector'] = target['Detector'].astype('int8')

# Split the target data into training and validation sets
target_train = target[target['DeviceId'].isin(device_id_train['DeviceId'])]
target_valid = target[target['DeviceId'].isin(device_id_valid['DeviceId'])]

# Save the target data to disk
target_train.to_csv('train/train_y.csv', index=False)
target_valid.to_csv('validate/val_y.csv', index=False)

# get distinct DeviceId, Detector and convert to Polars
# This is to filter to only include detectors that have labels
deviceid_detector = target[['DeviceId', 'Detector']].drop_duplicates()
deviceid_detector = pl.DataFrame(deviceid_detector)



def prepare_event_data(df, deviceid_detector, max_parameter=128):
    '''
    df: polars.DataFrame with hi-res event data (columns: DeviceId, Timestamp, EventId, Parameter)
    deviceid_detector: polars.DataFrame with distinct DeviceId and Detector columns to filter detectors
    max_parameter: maximum detector number to keep (default 128)
    '''
    # Filter and Cast EventId and Parameter to smaller integers for memory efficiency
    df = df\
        .filter(
            (pl.col('EventId').is_in([1, 7, 43, 44, 81, 82])) &
            (pl.col('Parameter') <= max_parameter) # Keep detector numbers below max_parameter because ODOT uses above 64 as dummy detectors
        )\
        .with_columns([
        pl.col('EventId').cast(pl.Int8),
        pl.col('Parameter').cast(pl.Int8)
    ])

    # Create separate dataframes for detector, phase green, and phase call events
    detector_events = (
        df.filter(pl.col('EventId').is_in([81, 82]))
        .with_columns(pl.col('Parameter').alias('Detector'))
        .join(deviceid_detector, on=['DeviceId', 'Detector'], how='inner')
    )

    phase_events = df.filter(pl.col('EventId').is_in([1, 7, 43, 44]))

    # Add events to set initial states to 0 for phases (first get min timestamp)
    min_timestamp = df['Timestamp'].min()
    
    # Get unique DeviceIds
    unique_devices = df.select('DeviceId').unique()

    # Create initial phase events for all phases (1-9) for each DeviceId
    initial_phase_events = []
    for phase in range(1, 10):
        # Create base dataframe with common columns for this phase
        base_df = unique_devices.with_columns([
            pl.lit(min_timestamp).alias('Timestamp'),
            pl.lit(phase).cast(pl.Int8).alias('Parameter')
        ])
        # Create events for both EventId 7 (phase green off) and 44 (phase call off)
        phase_off_df = base_df.with_columns([pl.lit(7).cast(pl.Int8).alias('EventId')])
        phase_call_off_df = base_df.with_columns([pl.lit(44).cast(pl.Int8).alias('EventId')])
        # Add both events to the list
        initial_phase_events.append(phase_off_df)
        initial_phase_events.append(phase_call_off_df)


    # Combine all initial phase events and reorder columns
    print('combining initial phase events')
    initial_phase_events = pl.concat(initial_phase_events).select(['DeviceId', 'Timestamp', 'EventId', 'Parameter'])

    # Concatenate the initial phase events with the existing phase events
    phase_events = pl.concat([phase_events, initial_phase_events])

    # Get unique detectors per DeviceId
    device_detectors = (
        detector_events
        .select(['DeviceId', 'Detector'])
        .unique()
    )

    # Expand phase events by joining with device_detectors
    print('expanding phase events')
    expanded_phases = (
        phase_events
        .join(device_detectors, on='DeviceId', how='inner')
    )

    # Combine detector events with expanded phase events
    print('combining detector and phase events')
    combined_df = (
        pl.concat([
            detector_events,
            expanded_phases
        ])
        .sort(['DeviceId', 'Detector', 'Timestamp', 'EventId'])
    )

    # Create state columns using window functions
    print('creating state columns')
    result_df = (
        combined_df
        .with_columns([
            # Set detector state
            pl.when(pl.col('EventId') == 82).then(1.0)
            .when(pl.col('EventId') == 81).then(0.0)
            .otherwise(None)
            .cast(pl.Float32)
            .alias('DetectorState')
        ])
    )

    # Add phase columns
    print('adding phase columns')
    # Add 2-3 columns at a time instead of all 18 at once
    for phase in range(1, 10):
        print(f'Adding phase {phase} columns')
        result_df = result_df.with_columns([
            # Add PhaseGreen
            pl.when((pl.col('Parameter') == phase) & (pl.col('EventId') == 1)).then(1.0)
            .when((pl.col('Parameter') == phase) & (pl.col('EventId') == 7)).then(0.0)
            .otherwise(None)
            .cast(pl.Float32)
            .alias(f'PhaseGreen{phase}'),

            # Add PhaseCall
            pl.when((pl.col('Parameter') == phase) & (pl.col('EventId') == 43)).then(1.0)
            .when((pl.col('Parameter') == phase) & (pl.col('EventId') == 44)).then(0.0)
            .otherwise(None)
            .cast(pl.Float32)
            .alias(f'PhaseCall{phase}')
        ])
        print(f'Added phase {phase} columns')

    # Forward fill the state columns (the last known state is carried forward)
    print('forward filling state columns')
    fill_columns = ['DetectorState'] + \
        [f'PhaseGreen{phase}' for phase in range(1, 10)] + \
        [f'PhaseCall{phase}' for phase in range(1, 10)]
    #result_df = result_df.with_columns([
    #    pl.col(col).forward_fill().over(['DeviceId', 'Detector']) for col in fill_columns
    #])
    # Forward fill columns in smaller groups
    for i in range(0, len(fill_columns), 3):  # Process 3 columns at a time
        print(f'Processing columns {i} to {i+2}')
        columns_chunk = fill_columns[i:i+3]
        result_df = result_df.with_columns([
            pl.col(col).forward_fill().over(['DeviceId', 'Detector'])
            for col in columns_chunk
        ])
        print(f'Filled columns: {columns_chunk}')


    # Calculate time deltas
    print('calculating time deltas')
    result_df = (
        result_df
        .with_columns([
            (pl.col('Timestamp').diff().over(['DeviceId', 'Detector']) / pl.duration(nanoseconds=1_000_000_000))
            .cast(pl.Float32)
            .alias('Delta')
        ])
        # Select and arrange final columns
        .select([
            'DeviceId',
            'Detector',
            'Timestamp',
            *[f'PhaseGreen{phase}' for phase in range(1, 10)],
            *[f'PhaseCall{phase}' for phase in range(1, 10)],
            'DetectorState',
            'Delta'
        ])
        .drop_nulls()
    )

    return result_df.sort(['DeviceId', 'Detector', 'Timestamp'])

# Process each day of training data and save to disk
for i in range(2, 5):
    print(f'Processing day {i}')
    df = pl.read_parquet(f'train/Train_Dec_{i}_2024.parquet')
    result_df = prepare_event_data(df, deviceid_detector, max_parameter=64)
    # Filter to training DeviceIds & save to disk
    result_df.filter(pl.col('DeviceId').is_in(device_id_train['DeviceId'])).write_parquet(f'train/train_X_{i}.parquet')
    # Filter to validation DeviceIds & save to disk
    result_df.filter(pl.col('DeviceId').is_in(device_id_valid['DeviceId'])).write_parquet(f'validate/val_X_{i}.parquet')