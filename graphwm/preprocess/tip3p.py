import numpy as np
import os
from pathlib import Path
from graphwm.data.utils import store_data
from random import sample
'''
This file save water tip3p data to h5 format that is consistent with mlcgmd project. It also create splits files for train validation and test split. 

'''



tip3p_posScale = 1
tip3p_energyScale = 1
# tip3p_forceScale = 1*(kilojoules_per_mole / nanometer)/(kilojoules_per_mole /angstrom)
tip3p_forceScale = 0.1

def load_water_tip3p(data_path, cell_size):
    particle_type = []
    for i in range(258 * 3):
        particle_type.append(21 if i % 3 == 0 else 10)  # 21: O, 10: H
    with np.load(data_path + '.npz', 'rb') as raw_data:
        pos = np.array(raw_data['pos']).reshape(-1, 3) * tip3p_posScale
        particle_type = np.array(particle_type).reshape(-1)    # atom type
        assert particle_type.shape[0] == pos.shape[0]
        e = np.array(raw_data['energy']).reshape(1,-1) * tip3p_energyScale
        f = np.array(raw_data['forces']).reshape(-1, 3) * tip3p_forceScale
        pos = np.remainder(pos, cell_size)
        assert 258*3 == pos.shape[0]
    return particle_type, pos, e, f

def save_water_h5(seed, positions, forces, energies, ptypes, bonds, save_path):
    poly_index = str(seed)
    os.makedirs(os.path.join(save_path, poly_index), exist_ok=True)
    if not Path(str(os.path.join(save_path, poly_index, 'bond.h5'))).exists():
      try:
        store_data(['position'], [positions], os.path.join(save_path, poly_index, 'position.h5'))
        store_data(['force'], [forces], os.path.join(save_path, poly_index, 'force.h5'))
        store_data(['particle_type'], [ptypes], os.path.join(save_path, poly_index, 'ptype.h5'))
        store_data(['bond_indices'], [bonds], os.path.join(save_path, poly_index, 'bond.h5'))
        store_data(['energy'], [energies], os.path.join(save_path, poly_index, 'energy.h5'))
      except Exception as e:
        print(poly_index)
        print(e)
        pass

def save_water():
    data_dir = '/home/stevenzhang/mdrefine/md_dataset/water_data_tip3p/'
    save_dir = '/home/stevenzhang/mlcgmd/water_data_tip3p/'

    cell_size = np.array([20,20,20]).reshape(1, 3)

    water_bonds = np.array([[0,1], [1,2], [0,2]]) # hard code water bonds
    bonds = []
    for i in range(258):
        bonds.append(water_bonds+i*3)
    bonds = np.array(bonds).reshape(-1,2)

    for x in range(100):
        positions = []
        energies = []
        forces = []

        for y in range(1000):
            fname = 'data_{}_{}'.format(x, y)
            data_path = os.path.join(data_dir, fname)
            particle_type, pos, e, f = load_water_tip3p(data_path, cell_size = cell_size)
            positions.append(pos)
            energies.append(e)
            forces.append(f)
        save_water_h5(seed=x,positions=positions,forces=forces, energies=energies, ptypes=particle_type, bonds=bonds,save_path=save_dir)


def generate_splits(train_ratio, val_ratio, test_ratio, seed_num, save_dir):
    assert train_ratio+val_ratio+test_ratio==1
    total_numbers = sample(range(seed_num), seed_num)  # 100 unique numbers
    x_numbers, y_numbers, z_numbers = total_numbers[:train_ratio*seed_num], total_numbers[train_ratio*seed_num:(train_ratio+val_ratio)*seed_num], total_numbers[(train_ratio+val_ratio)*seed_num:]

    splits = ['train','val', 'test']
    for i, numbers in enumerate([x_numbers, y_numbers, z_numbers], start=1):
        file_path = os.path.join(save_dir,f'{splits[i]}.txt')
        with open(file_path, 'w') as file:
            for number in numbers:
                file.write(f'{number}\n')

# generate_splits(0.9, 0.05, 0.05, 100, './')
