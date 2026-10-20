"""Render one part with a plugin in this interpreter (run under Rosetta for Intel-only plugins such as Vital).

Usage (called by instmate/vst.py): arch -x86_64 .venv-x86/bin/python scripts/vst_worker.py job.json out.npy
job.json: {"plugin": path, "state": state file or null, "messages": [[type, note, velocity, control, value, time_s]],
           "duration": seconds, "sample_rate": 44100}
"""
import json
import os
import sys

import numpy as np


def main(job_path, out_path):
    import pedalboard
    from mido import Message
    job = json.load(open(job_path))
    plug = pedalboard.load_plugin(os.path.expanduser(job["plugin"]))
    if job.get("state") and os.path.isfile(job["state"]):
        plug.raw_state = open(job["state"], "rb").read()
    msgs = []
    for kind, note, vel, control, value, t in job["messages"]:
        if kind == "control_change":
            msgs.append(Message(kind, control=control, value=value, time=t))
        else:
            msgs.append(Message(kind, note=note, velocity=vel, time=t))
    audio = plug(msgs, duration=job["duration"], sample_rate=job["sample_rate"], num_channels=2)
    np.save(out_path, np.asarray(audio, dtype=np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
