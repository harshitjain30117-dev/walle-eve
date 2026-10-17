# Mission report: WALL-E and EVE's Signal

Agent: **greedy** | seed: 12 | map 40x30 | decoy probability 0.15

## Outcome

- **Inferred location:** (38, 2)
- **Inferred status:** active (baseline does not model status)
- **Direct detection by scan:** no
- **Resources:** 240/240 time, 198 battery spent, 188 moves, 2 scans, 8 recharges, final battery 38

## Evidence found

- Initial signal: bearing 355 deg, strength 1.62 at the base.
- Step 32: `debris` clue at (21, 12): EVE about 11.5 cells away.
- Step 42: `debris` clue at (26, 9): EVE about 6.5 cells away.

## Strategy changes

- None (single plan).

## Recommended next step

- EVE has **not** been confirmed. Best hypothesis: (38, 2) (active), confidence nan.
- Next step: scan around (38, 2), 23 moves from WALL-E's current cell.
- Recharge at a station before the leg if battery is below the round-trip need of 57.

## Ground-truth check (simulation only)

- True position (28, 2), true status dormant.
- Error 10.0 cells (tolerance 3.0): location WRONG, status WRONG.
- Hidden decoys at [(18, 15), (14, 1)]; 2/6 clues discovered.
