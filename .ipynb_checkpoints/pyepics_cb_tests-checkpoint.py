__all__ = """
    motors
    update_model
    clear_motor_cbs
    clear_simulation_cbs
    install_simulation_cbs
""".split()

from epics import PV

prefix = '100idWYM'
motor_list = ['m1', 'm2', 'm3', 'm4', 'm5']
pv_list = [':'.join([prefix,motor]) for motor in motor_list]

motors = [PV(motor_pv) for motor_pv in pv_list]

def update_model(pvname = None, value = None, **kw):
    print('PV changed: ', pvname,' + ', value)
    for pvs in motors:
        if pvname != pvs.pvname:
            print(pvs.pvname,' is still at ',pvs.get())
            
def clear_motor_cbs(motors):
    for motor in motors:
        motor.clear_callbacks()
        
def clear_simulation_cbs(motors, indices):
    for motor, index in zip(motors, indices):
        motor.remove_callback(index=index)
        
def install_simulation_cbs(motors, func):
    indices = []
    for motor in motors:
        indices.append(motor.add_callback(callback=func))
    return indices
    
