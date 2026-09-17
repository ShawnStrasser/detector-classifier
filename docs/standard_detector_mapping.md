# Standard detector channel -> phase wiring (332 / 332S cabinets)

Transcribed 2026-09-17 from the table supplied by Shawn. The channel -> phase default is the same for both
cabinet types; they differ in the input-file slot and in that only the 332S has channels 29-40.

| Det | Phase | 332 slot | 332S slot | | Det | Phase | 332 slot | 332S slot |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | I1U | I1U | | 21 | 7 | J5U | J6U |
| 2 | 2 | I2U | I3U | | 22 | 8 | J6U | J8U |
| 3 | 2 | I2L | I3L | | 23 | 8 | J6L | J8L |
| 4 | 2 | I3U | I4U | | 24 | 8 | J7U | J9U |
| 5 | 2 | I3L | I4L | | 25 | 8 | J7L | J9L |
| 6 | 2 | I4U | I5U | | 26 | 8 | J8U | J10U |
| 7 | 3 | I5U | I6U | | 27 | 5 | J9U | J1L |
| 8 | 4 | I6U | I8U | | 28 | 7 | J9L | J6L |
| 9 | 4 | I6L | I8L | | 29 | 1 | - | I2U |
| 10 | 4 | I7U | I9U | | 30 | 1 | - | I2L |
| 11 | 4 | I7L | I9L | | 31 | 2 | - | I5L |
| 12 | 4 | I8U | I10U | | 32 | 3 | - | I7U |
| 13 | 1 | I9U | I1L | | 33 | 3 | - | I7L |
| 14 | 3 | I9L | I6L | | 34 | 4 | - | I10L |
| 15 | 5 | J1U | J1U | | 35 | 5 | - | J2U |
| 16 | 6 | J2U | J3U | | 36 | 5 | - | J2L |
| 17 | 6 | J2L | J3L | | 37 | 6 | - | J5L |
| 18 | 6 | J3U | J4U | | 38 | 7 | - | J7U |
| 19 | 6 | J3L | J4L | | 39 | 7 | - | J7L |
| 20 | 6 | J4U | J5U | | 40 | 8 | - | J10L |

As a lookup table alone this matches 92.6% of training labels (91.6% statewide); 273 of 418 signals match 100%.

```python
DEFAULT_PHASE = dict(zip(range(1, 41),
    [1,2,2,2,2,2,3,4,4,4,4,4,1,3,5,6,6,6,6,6,7,8,8,8,8,8,5,7,1,1,2,3,3,4,5,5,6,7,7,8]))
```
