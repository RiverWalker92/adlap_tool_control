# How to run an automated SATA measurement

This guide describes the operator workflow implemented in `adlap_tool_control`: start the controller, identify the hardware, complete coupling and initialisation, confirm readiness, and let the automated runner execute the configured DoFs and post-processing.

The instructions were checked against the supplied source code and configuration. A live hardware run was not performed during this review. They assume the existing lab ROS 2 workspace is installed, built and connected to the SATA hardware. Follow the established mechanical assembly and calibration procedure for the particular gearbox/instrument.

## Quick overview

| Stage | Terminal | Operator or automatic? |
| --- | --- | --- |
| Start the right controller | A | Operator |
| Start `automated_trial_runner.py` | B | Operator |
| Show the hardware markers and finish scanning | Camera window and B | Operator, with automatic marker recognition |
| Couple the hardware, initialise and update starting positions | A and the physical setup | Operator using the existing controller routines |
| Confirm that preparation is complete | B | Operator presses Enter |
| Execute DoF1, DoF3, DoF4 and DoF2 | B and child ROS launches | Automatic, in the configured order |
| Log measurements, replay the Motor DT, process supported video, plot and run diagnostics | B and child processes | Automatic after each DoF |

**Initialisation is an operator-controlled preparation step. The trial runner pauses for it; the runner does not itself press the controller keys or perform the physical coupling.**

## 1. Check the setup and sequence settings

Use the intended hardware and make its markers available to the webcam. Connect the controller hardware and ensure that the workspace and required model files are available on this computer.

Check `~/ros2_ws/src/adlap_tool_control/config/dof_pattern_params.yaml`. The supplied configuration contains:

```yaml
dof_pattern_runner_node:
  ros__parameters:
    run_all_dofs: true
    dof_sequence_order: [1, 3, 4, 2]
```

This means:

| Order | DoF | Movement |
| --- | --- | --- |
| 1 | DoF1 | Axial roll |
| 2 | DoF3 | Tip rotation |
| 3 | DoF4 | Jaw opening/closing |
| 4 | DoF2 | Bending |

The waveform, range, frequency, cycles and pauses are defined in the corresponding `dof1` to `dof4` entries. Retain the settings used for each experiment. A command-line `--dof` selects a single DoF and bypasses this outer all-DoF sequence.

The standard guide below uses the existing source/configuration paths. Although the parent runner accepts `--params-file`, its current child launch file also refers to the standard source-tree YAML path. For a changed protocol, keep the actual configuration loaded by the launch consistent; do not assume that an arbitrary alternate parent YAML is automatically forwarded everywhere.

## 2. Terminal A: start the controller

Open a terminal in the configured ROS environment. If the workspace environment has not been loaded, source it:

```bash
source ~/ros2_ws/install/setup.bash
```

The appropriate ROS 2 distribution/underlay must also be available. Use the existing lab installation rather than guessing a different distribution.

Start the right controller in this terminal:

```bash
ros2 run adlap_tool_control controller --ros-args -r __ns:=/right
```

This is the controller executable/namespace used by the supplied `right.launch.py`. Starting it directly keeps the keyboard interface in Terminal A. The runner expects `/right/tool_control_node`, including its `enable_bend_play_compensation` parameter.

Leave the controller running. Do not start a second controller for the same right-hand setup. If the normal lab launch already provides this controller terminal, use that existing controller instead.

## 3. Terminal B: start the automated trial runner

Open a second terminal in the same ROS/workspace environment:

```bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/adlap_tool_control/scripts/runners/automated_trial_runner.py
```

Do not add `--dof` when you want the complete configured DoF sequence. The supplied runner's current `auto` mode resolves to DoF patterns, including for motor-only operation.

The runner first applies camera settings and starts the ArUco detector. Defaults in the supplied runner are `/dev/video0`, requested 3840 x 2160 resolution, 30 fps, focus 150 and sharpness 150. Actual supported camera properties depend on the webcam; the runner prints the recording frame size.

## 4. Identify the hardware

Show the relevant markers one at a time. A known marker is accepted after stable detection. Answer `y` to `Scan another marker? [y/n]:` when another component still needs to be scanned. Finish scanning when all required components have been recognised.

The supplied `config/marker_detection.yaml` maps:

| Marker ID | Component | Configuration identity |
| --- | --- | --- |
| 10 | Old blue gearbox | gearbox_1 |
| 11 | Long-collet gearbox | gearbox_2 |
| 20 | Black shaft | shaft_black |
| 21 | Silver shaft A | shaft_A |
| 30 | Gripper tip | gripper |
| 31 | Scissors tip | scissors |

The accepted combinations are:

| Hardware | Required recognised components | coupling_mode |
| --- | --- | --- |
| Motor unit only | No gearbox, shaft or tip markers | motor_only |
| Motor unit and gearbox | One gearbox | gearbox_only |
| Full system | One gearbox, one shaft and one tip | full_setup |

For motor-only identification, finish the scan without accepting component markers (`d` in the camera window). For a full setup, do not finish after scanning only the gearbox or only part of the instrument. Multiple markers of the same component type and incomplete combinations are rejected by the classifier.

The detected configuration is printed and saved as `_latest_detected_hardware.json` under the selected automated-trials output root. Check that the displayed configuration matches the physical setup.

## 5. Complete coupling and initialisation in Terminal A

For `gearbox_only` and `full_setup`, Terminal B now displays a preparation pause and asks you to return to the controller terminal. Complete physical coupling and the existing initialisation procedure before confirming readiness.

The supplied controller's manual keyboard interface provides the following actions. Use lower-case keys; the console labels may show upper-case letters.

| Key in Terminal A | Implemented action |
| --- | --- |
| `c` | Run the gearbox coupling sequence, when required by the mechanical procedure |
| `i` | Run motor initialisation (`setup_motors`) |
| `u` | Update the starting positions after initialisation |
| Enter | Finish the manual keyboard interface and update starting positions once more |

Use the appropriate coupling/initialisation order for the actual physical assembly, then complete `i`, `u` and the final Enter where required by the established procedure. Wait for each routine to finish before proceeding. The `u` action is not merely a readout: the supplied code also moves M3 to apply jaw-closing tension before recording the start positions.

The controller rejects `u` and the final manual-interface Enter if its local initialisation state has not been set by `i`.

**There are two different Enter actions:** Enter in Terminal A finishes the manual controller interface; Enter in Terminal B releases the automated runner's preparation pause. Complete Terminal A first.

For `motor_only`, the runner skips this preparation pause. Prepare the motor-only setup using its established procedure before starting the runner. Do not apply gearbox-dependent coupling/calibration routines to an uncoupled motor unit simply to reproduce the coupled-setup key sequence.

## 6. Terminal B: confirm readiness

Return to Terminal B after coupling, initialisation and the start-position update are complete. At the prompt:

```text
Press Enter here only when coupling, initialization, and the start-position update are done...
```

press Enter.

The runner then sets `enable_bend_play_compensation` on `/right/tool_control_node`. The supplied source explicitly sets it to `true` before the sequence. The controller must therefore already be running in the expected namespace.

With `run_all_dofs: true` and no `--dof` override, the runner starts:

```text
DOF1 -> DOF3 -> DOF4 -> DOF2
```

It scans the markers and asks for coupled-setup preparation once. Child processes reuse the selected configuration; they do not repeat the scan or the preparation pause between DoFs.

## 7. What happens automatically for each DoF?

For each selected DoF, the runner:

1. Creates a timestamped run folder and filenames.
2. Starts video recording when enabled for that configuration/DoF.
3. Launches the DoF pattern runner, ROS logger and the applicable gearbox/instrument state nodes.
4. Executes the configured waveforms while logging commands, measured signals, state predictions, task labels and synchronisation information.
5. Waits for the pattern launch to finish and stops recording.
6. Runs the offline Motor DT replay with the matching coupling mode, gearbox/instrument identity and active DoF.
7. Produces motor-output plots and, where applicable, instrument-output plots and video-angle measurements.
8. Writes metadata, then runs the automatic diagnostic script unless `--skip-diagnostics` was requested.
9. Starts the next DoF only after the current child process completes successfully.

If a child process returns an error, the parent stops the sequence and identifies the failed DoF. Earlier run folders can still exist even when the complete sequence did not finish. A successful sequence ends with `ALL DOF TESTS COMPLETED`.

The automatic diagnostics are the `digital_twin_diagnostics.py` workflow. This acquisition command does not run the separate multi-recording LDA analyses or the healthy-model LOOCV benchmark.

## 8. Video behaviour and model selection

Video measurements are used for supported DoF2/DoF4 full-system trials. The supplied standard scissors DoF4 route disables recording/angle detection; the dedicated cutting-test route has different logic. Use `--no-camera` explicitly for cutting measurements when angle extraction is not intended.

After successful angle extraction, the current runner **deletes the temporary raw `*_webcam.mp4` recording**. The angle detector can generate processed/debug video and image outputs. Do not assume that the original raw video is retained simply because a video path appears in metadata. Preserving original videos for future reprocessing requires an explicit change to that deletion behaviour before acquiring the measurements.

Motor DT replay loads stored configuration-specific models; it does not train them. Confirm that those packages exist at their configured/default locations.

The full-system instrument node also uses `config/instrument_state_node.yaml`. In the supplied copy, `hybrid_bend_enabled` is true and a nonempty `hybrid_bend_model_file` explicitly points to an older V6 Ridge model. That explicit path takes precedence over automatic selection from the training configuration. Check and record the intended model path and the `Hybrid Bend inputs` printed by the node. Do not describe a new automated acquisition as using the verified six-input thesis model until that actual runtime selection has been checked.

The verified offline six-input benchmark is a separate runnable package. Creating this how-to has not changed the controller, its parameters, the runtime model selection or any source code.

## 9. Locate and check the generated files

The default output root is:

```text
~/ros2_ws/test_data/automated_trials/
```

| Configuration | Folder pattern below that root |
| --- | --- |
| Motor-only | setup_01_motors/dofN/auto_dofN_TIMESTAMP/ |
| Motor-gearbox | setup_02_motors_gearbox/GEARBOX/dofN/auto_dofN_TIMESTAMP/ |
| Full system | setup_03_motors_gearbox_instrument/GEARBOX/INSTRUMENT/dofN/auto_dofN_TIMESTAMP/ |

Each DoF is a separate run with its own timestamp. Expected outputs include:

| Output | Meaning |
| --- | --- |
| `*_ros_log.jsonl` | Original ROS recording for this DoF |
| `*_motor_dt_replay.jsonl` | Offline Motor DT predictions and residuals |
| `*_metadata.json` | Hardware identity, paths, commands, active DoF and processing choices |
| `*_webcam_angles.jsonl` | Video-angle measurements, when the video route was used |
| `plots/motor_outputs/` | Motor plots |
| `plots/instrument_outputs/` | Applicable instrument plots |
| `diagnostics/digital_twin_diagnosis.json` | Diagnostic result when the default diagnostic output directory is used |

Check the four DoF folders and metadata before treating the measurement sequence as complete. Retain each source log, replay and applicable angle log together. Keep the exact configurations and software revision used for the acquisition.

## 10. Common variations

Run one standard DoF:

```bash
python3 ~/ros2_ws/src/adlap_tool_control/scripts/runners/automated_trial_runner.py --dof 2
```

Run the dedicated DoF3 resistance protocol:

```bash
python3 ~/ros2_ws/src/adlap_tool_control/scripts/runners/automated_trial_runner.py --dof 3 --resistance-test
```

Run the dedicated DoF4 cutting protocol without angle-video processing:

```bash
python3 ~/ros2_ws/src/adlap_tool_control/scripts/runners/automated_trial_runner.py --dof 4 --cutting-test --no-camera
```

The resistance flag requires `--dof 3`; the cutting flag requires `--dof 4`. They cannot be combined.

If identification is deliberately specified manually rather than scanned, use all required identifiers, for example:

```bash
python3 ~/ros2_ws/src/adlap_tool_control/scripts/runners/automated_trial_runner.py \
  --skip-marker-scan \
  --coupling-mode full_setup \
  --gearbox-variant gearbox_2 \
  --instrument-config gripper
```

`--no-camera` disables measurement recording/angle processing, but does not itself skip the initial marker scan. For operation without a camera at all, also use the explicit manual-identification options above and `--no-camera`.

Use the normal defaults for the standard complete sequence. Available switches can be inspected without launching a measurement:

```bash
python3 ~/ros2_ws/src/adlap_tool_control/scripts/runners/automated_trial_runner.py --help
```

## Troubleshooting and scope

| Symptom | First check |
| --- | --- |
| Parameter set fails at `/right/tool_control_node` | Controller running and workspace/namespace correct |
| Only one DoF runs | `--dof` supplied, `run_all_dofs` disabled, or sequence stopped after an error |
| Wrong DoF order | `dof_sequence_order` in the configuration actually used |
| Marker combination rejected | Exactly one required gearbox/shaft/tip combination scanned |
| `u` or Enter rejected in controller terminal | Complete the controller's `i` initialisation first |
| Replay/model loading fails | Expected model file exists and compatible numerical environment is active |
| Bending predictions use an unexpected model | Nonempty `hybrid_bend_model_file` override in instrument_state_node.yaml |
| Raw webcam file is absent after success | Current runner deletes that temporary raw video after angle extraction |

This how-to describes the automation and operator interface in the supplied source. Complete ROS installation/build instructions, hardware-specific assembly details and a live acceptance test remain separate from this source-level documentation check.
