import os
import pygame
import torch
import numpy as np

from lib.displayer import displayer
from lib.utils import mat_distance
from lib.CAModel import CAModel
from lib.utils_vis import to_rgb, make_seed

eraser_radius = 6

pix_size = 8
display_map_shape = (120, 120)
_map_shape = (80, 80)
CHANNEL_N = 16
CELL_FIRE_RATE = 0.5
model_path = "models/test2.pth"
device = torch.device("cpu")

torch.set_grad_enabled(False)

_rows = np.arange(_map_shape[0]).repeat(_map_shape[1]).reshape([_map_shape[0],_map_shape[1]])
_cols = np.arange(_map_shape[1]).reshape([1,-1]).repeat(_map_shape[0],axis=0)
_map_pos = np.array([_rows,_cols]).transpose([1,2,0])

_map = make_seed(_map_shape, CHANNEL_N)
seed_values = np.zeros((2, CHANNEL_N - 3))
seed_values[:, 0] = 1
seed_values[1, :int((CHANNEL_N - 3)/2)] = 1
seed_values[0, int((CHANNEL_N - 3)/2)+1:] = 1
curr_target = 0

print(seed_values.shape)
print(_map.shape)
_map[_map.shape[0]//2, _map.shape[1]//2, 3:] = seed_values[curr_target]
display_map = np.ones([*display_map_shape, 3]) -0.00001

model = CAModel(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
output = model(torch.from_numpy(_map.reshape([1,_map_shape[0],_map_shape[1],CHANNEL_N]).astype(np.float32)), 1)

disp = displayer(display_map_shape, pix_size)

isMouseDown = False
isSpaceDown = False
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:

            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                isMouseDown = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                isMouseDown = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                isSpaceDown = True

    if isMouseDown:
        try:
            mouse_pos = np.array([int(event.pos[1]/pix_size - 20), int(event.pos[0] / pix_size - 20)])
            should_keep = (mat_distance(_map_pos, mouse_pos)>eraser_radius).reshape([_map_shape[0],_map_shape[1],1])
            output = output * torch.tensor(should_keep)
        except AttributeError:
            pass
    elif isSpaceDown:
        curr_target = (curr_target + 1) % 2
        output = make_seed(_map_shape, CHANNEL_N)
        output[_map.shape[0]//2, _map.shape[1]//2, 3:] = seed_values[curr_target]
        output = torch.from_numpy(output[np.newaxis])
        isSpaceDown = False

    output = model(output, 1)

    _map = to_rgb(output.numpy()[0])
    display_map[20:100, 20:100] = _map
    disp.update(display_map)
