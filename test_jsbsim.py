import jsbsim
import os

root_dir = os.path.dirname(jsbsim.__file__)
print(f"JSBSim root: {root_dir}")

sim = jsbsim.FGFDMExec(root_dir, log_lvl=0)

try:
    sim.load_model('c172p')
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Load failed: {e}")
    exit()

# Set initial conditions
sim['ic/h-sl-ft'] = 1000.0
sim['ic/long-gc-deg'] = -122.0
sim['ic/lat-gc-deg'] = 36.0
sim['ic/ubody-fps'] = 80.0
sim['ic/vbody-fps'] = 0.0
sim['ic/wbody-fps'] = 0.0
sim['ic/phi-rad'] = 0.0
sim['ic/theta-rad'] = 0.0
sim['ic/psi-rad'] = 0.0

try:
    sim.run_ic()
    print("✅ Initial conditions applied")
except Exception as e:
    print(f"❌ run_ic failed: {e}")

print(f"Altitude: {sim['position/h-sl-ft']} ft")
print(f"Speed: {sim['velocities/ve-fps']} fps")

