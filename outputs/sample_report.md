# Mission report: WALL-E and EVE's Signal

Agent: **bayes** | seed: 12 | map 40x30 | decoy probability 0.15

## Outcome

- **Inferred location:** (28, 2)
- **Inferred status:** dormant (P(active)=0.00, P(dormant)=1.00)
- **Confidence:** 1.00 (posterior mass within 2.0 cells of the estimate)
- **Residual entropy:** 0.00 nats (start: 6.64)
- **Direct detection by scan:** yes
- **Resources:** 139/240 time, 133 battery spent, 99 moves, 8 scans, 4 recharges, final battery 37

## Evidence found

- Initial signal: bearing 355 deg, strength 1.62 at the base.
- Step 59: `debris` clue at (21, 12): EVE about 11.5 cells away.
- Step 69: `debris` clue at (26, 9): EVE about 6.5 cells away.
- Step 111: **scan detected EVE** at (28, 2), reported status dormant.
- Spoof check: 18 of 108 signal readings are inconsistent (> 3 sigma) with the final estimate and are treated as suspected decoys.

## Strategy changes

- Step 2 (t=3, battery 55): retargeted (2, 15) -> (18, 9) (trigger: plan complete)
- Step 7 (t=9, battery 47): retargeted (21, 6) -> (7, 15) (trigger: belief shift)
- Step 8 (t=10, battery 46): retargeted (7, 15) -> (24, 6) (trigger: plan complete)
- Step 10 (t=12, battery 44): retargeted (24, 6) -> (9, 12) (trigger: belief shift)
- Step 24 (t=30, battery 18): mode explore -> return (trigger: plan complete)
- Step 37 (t=48, battery 36): mode return -> recharge (trigger: plan complete)
- Step 39 (t=55, battery 59): mode recharge -> explore (trigger: plan complete)
- Step 74 (t=91, battery 19): mode explore -> return (trigger: plan complete)
- Step 90 (t=112, battery 34): mode return -> recharge (trigger: plan complete)
- Step 92 (t=119, battery 59): mode recharge -> explore (trigger: plan complete)

## Recommended next step

- EVE is confirmed at (28, 2) (dormant). Dispatch WALL-E: the target is 2 moves from its current cell (27, 3), 21 moves from the nearest charging station.
- Battery 37 covers the round trip (26 needed incl. reserve); go now.
- EVE appears **dormant** (weak emitter): bring a power-transfer / wake-up routine and expect a faint approach signal.

## Ground-truth check (simulation only)

- True position (28, 2), true status dormant.
- Error 0.0 cells (tolerance 3.0): location correct, status correct.
- Spoofed readings actually received: 17; flagged correctly: 17; false alarms: 1.
- Hidden decoys at [(18, 15), (14, 1)]; 2/6 clues discovered.
