# ForceRL


# Depandencies
Some depandencies are in environment.yml file

# Fixing issue for rendering

type this in terminal before running code to get around OpenGL issue

```export MUJOCO_GL="osmesa"```

To permenantly get rid of this issue, add the above line to your ```.bashrc``` file. 

# Environment TODOs

`original_door_env.py` contains the original door environment copied from robosuite. It contains just a door without robot.

`curri_door_env.py` contains the door environment for curricula of door pose initialized in increasing range. It extends original door env.

`door_random_point_env.py` contains the door environment that generates random force point each episodes. It extends curriclum door env. 



# Findings

Curriculum learning with 

```
fixed initial pose -> wider initial pose -> random point
```

works good.

Do we need discretization if so?



# Revolute object Axis:
For microwave, X is facing outward, Y is facing up, Z is facing "left"
dishwasher: X facing outward, Y facing downward, Z facing "right"
Door counterclock: X out, Y left, Z down 
