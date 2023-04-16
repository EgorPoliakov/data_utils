from enum import IntEnum

class Action(IntEnum):
    PULL_UP = 0,
    PUSH_UP = 1,
    JUMP_ROPE = 2,
    REST = 3

# action_id_to_name = {
#     'pull_up': Action.PULL_UP
#     'push_up': Action.PUSH_UP
#     'jumping_rope': Action.JUMPING_ROPE
#     'rest': Action.REST
# }