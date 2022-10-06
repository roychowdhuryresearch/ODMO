import numpy as np
working_dir= "."
mocap_raw_offsets = np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 1, 0],
                             [0, 1, 0],
                             [0, 1, 0], 
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 1, 0],
                             [0, 1, 0],
                             [-1, 0, 0],
                             [-1, 0, 0],
                             [-1, 0, 0],
                             [0, -1, 0],
                             [0, -1, 0],
                             [0, -1, 0],
                             [0, -1, 0],
                             [0, -1, 0],
                             [0, -1, 0],
                             [0, -1, 0],
                             [0, -1, 0]])

# Define a kinematic tree for the skeletal struture
mocap_kinematic_chain = [[0, 1, 2, 3], [0, 12, 13, 14, 15], [
    0, 16, 17, 18, 19], [1, 4, 5, 6, 7], [1, 8, 9, 10, 11]]

mocap_class_names = ["Walk", "Run", "Jump",
                     "Animal Behavior", "Dance", "Step", "Climb"]
uestc_kinematic_chain = [[0, 12, 13, 14, 15],
                        [0, 9, 10, 11, 16],
                        [0, 1, 8, 17],
                        [1, 5, 6, 7],
                        [1, 2, 3, 4]]
humanact12_class_names = ["Warm up", "Walk", "Run", "Jump", "Drink",
                          "Lift dumbbell", "Sit", "Eat", "Turn steering wheel", "Phone", "Boxing", "Throw"]
uestc_class_names = ['punching-and-knee-lifting',
 'marking-time-and-knee-lifting',
 'jumping-jack',
 'squatting',
 'forward-lunging',
 'left-lunging',
 'left-stretching',
 'raising-hand-and-jumping',
 'left-kicking',
 'rotation-clapping',
 'front-raising',
 'pulling-chest-expanders',
 'punching',
 'wrist-circling',
 'single-dumbbell-raising',
 'shoulder-raising',
 'elbow-circling',
 'dumbbell-one-arm-shoulder-pressing',
 'arm-circling',
 'dumbbell-shrugging',
 'pinching-back',
 'head-anticlockwise-circling',
 'shoulder-abduction',
 'deltoid-muscle-stretching',
 'straight-forward-flexion',
 'spinal-stretching',
 'dumbbell-side-bend',
 'standing-opposite-elbow-to-knee-crunch',
 'standing-rotation',
 'overhead-stretching',
 'upper-back-stretching',
 'knee-to-chest',
 'knee-circling',
 'alternate-knee-lifting',
 'bent-over-twist',
 'rope-skipping',
 'standing-toe-touches',
 'standing-gastrocnemius-calf',
 'single-leg-lateral-hopping',
 'high-knees-running']
mocap_action_enumerator = {
    0: "Walk",
    1: "Run",
    2: "Jump",
    3: "Animal Behavior",
    4: "Dance",
    5: "Step",
    6: "Climb"
}

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#3d1c02",
    "#0ffef9",
    "#ceaefa",
    "#280137"
]


# HumanAct12
HumanAct12_fine_classname_map = {"A0101": "warm_up_wristankle",
                                 "A0102": "warm_up_pectoral",
                                 "A0103": "warm_up_eblowback",
                                 "A0104": "warm_up_bodylean_right_arm",
                                 "A0105": "warm_up_bodylean_left_arm",
                                 "A0106": "warm_up_bow_right",
                                 "A0107": "warm_up_bow_left",
                                 "A0201": "walk",
                                 "A0301": "run",
                                 "A0401": "jump_handsup",
                                 "A0402": "jump_vertical",
                                 "A0501": "drink_bottle_righthand",
                                 "A0502": "drink_bottle_lefthand",
                                 "A0503": "drink_cup_righthand",
                                 "A0504": "drink_cup_lefthand",
                                 "A0505": "drink_both_hands",
                                 "A0601": "lift_dumbbell with _right hand",
                                 "A0602": "lift_dumbbell with _left hand",
                                 "A0603": "Lift dumbells with both hands",
                                 "A0604": "lift_dumbbell over head",
                                 "A0605": "lift_dumbells with both hands and bend legs",
                                 "A0701": "sit",
                                 "A0801": "eat_finger_right",
                                 "A0802": "eat_pie/hamburger",
                                 "A0803": "Eat with left hand",
                                 "A0901": "Turn steering wheel",
                                 "A1001": "Take out phone, call and put phone back",
                                 "A1002": "Call with left hand",
                                 "A1101": "boxing_left_right",
                                 "A1102": "boxing_left_upwards",
                                 "A1103": "boxing_right_upwards",
                                 "A1104": "boxing_right_left",
                                 "A1201": "throw_right_hand",
                                 "A1202": "throw_both_hands"}
HumanAct12_coarse_classname_map = {"A01": "Warm up", "A02": "Walk", "A03": "Run", "A04": "Jump", "A05": "Drink",
                                   "A06": "Lift dumbbell", "A07": "Sit", "A08": "Eat", "A09": "Turn steering wheel", "A10": "Phone", "A11": "Boxing", "A12": "Throw"}

humanact12_kinematic_chain = [[0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [
    0, 3, 6, 9, 12, 15], [9, 13, 16, 18, 20, 22], [9, 14, 17, 19, 21, 23]]
vibe_kinematic_chain = [[0, 12, 13, 14, 15], [
    0, 9, 10, 11, 16], [0, 1, 8, 17], [1, 5, 6, 7], [1, 2, 3, 4]]
