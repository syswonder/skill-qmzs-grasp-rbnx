# Hardware validation

## 2026-11-05 — grasp / pick on the ARM7AIX cell

- **Platform**: ARM7AIX 7-DoF arm + LinkerHand O6 dexterous hand (CAN, left),
  U2CAN serial adapter.
- **Perception**: fixed Orbbec Gemini 336L above the table, tabletop hand-eye
  homography (`config/handeye/calibration.json` in the deployment), fixed
  flange Z = 110 mm and fixed quaternion from
  `config/handeye/grasp_orientation.json`.
- **Software**: this repository (`qmzs_grasp`, the code released as v0.1.0),
  running in the ARM7AIX Robonix deployment.
- **Flow exercised**: `pick` (VLM locates the green cube) → hand open →
  hover at `target z + approach_height_m` → descend → close hand (O6 fist
  preset) → settle → lift back to hover. **The green cube was gripped and
  lifted off the table.** A 23 s / 1280×720 recording of the run was kept by
  the maintainers.
- **Outcome**: PASS for the grasp sequence (`pick`, and the shared
  open → approach → close → lift segment also used by `grab`).
- **Not covered by this run**: `place`, `release`, `home`, and multi-object
  or failure/retry paths. The O6 interface in this skill is open-loop
  (`hand_open_settle_s` / `hand_close_settle_s`, no `get_state` feedback), so
  operators should still watch the fingers/object during normal use.

## Note on earlier wording

The initial commit and the v0.1.0 PR descriptions said that real-hardware
grasp/place/home motion still required on-site sign-off. That statement was
written before this run was recorded; the grasp flow has since been validated
on the physical cell as described above. Place/release/home remain unvalidated
on hardware and are not claimed here.
