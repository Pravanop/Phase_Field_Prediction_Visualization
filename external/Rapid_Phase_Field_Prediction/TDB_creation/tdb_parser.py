import json

from pymatgen.core import Element
from utils.json_handler import JSONHandler
from utils.combination_generation import MultinaryCombinations
import numpy as np
import pickle


class TDBHandler:

	def __init__(self, composition, T_flag=True, im_flag=True, reg = False):

		self.T_flag = T_flag
		self.im_flag = im_flag
		self.SGTE = True
		self.reg = reg
		self.composition = composition
		self.eles = self.composition.split('-')
		self.load_unary_tdb()
		self.load_sgte_tdb()
		self.bokas_elements = JSONHandler.load_json("../database", "bokas_end_members_dict")
		self.lattice_info = {'FCC': 'FCC_A1', 'BCC': 'BCC_A2', 'HCP': 'HCP_A3', }
		self.transition_temperatures = {
			"Fe": ["BCC", "FCC", 1180],
			"Ti": ["HCP", "BCC", 1155],
			"Hf": ["HCP", "BCC", 2016],
			"Zr": ["HCP", "BCC", 1136],
			"Mn": ["BCC", "FCC", 1370],
		}
		self.reg_flag = False
		self.conversion = 96485
		if self.reg:
			self.binary_data = JSONHandler.load_json("../input/DFT_calculated", "bokas_omegas_processed")
		else:
			self.binary_data = JSONHandler.load_json("../input/DFT_calculated", "bokas_omegas_processed_subregular")

		with open(f"../database/intermetallics/im_list.pickle", 'rb') as handle:
			self.im_list = pickle.load(handle)

		self.file_writer()
		self.file_saver()

	def load_unary_tdb(self):
		with open('../input/Empirical_database/unary_tdb.json', 'r') as f:
			self.da = json.load(f)

	def load_sgte_tdb(self):
		with open('../input/Empirical_database/sgte.json', 'r') as f:
			self.sgteda = json.load(f)

	def topline(self):
		return f"$ {self.composition.upper().replace('-', '')}"

	def two_line(self):
		return "ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!\nELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!"

	def ele_line(self):
		return [self.da[i]['element'] for i in self.eles]

	def get_lattices(self, ele_line):
		return [i.split('   ')[1].replace(' ', '') for i in ele_line]

	def element_data(self, ans):

		ans += self.topline() + '\n'
		ans += self.two_line() + '\n'
		ele_line = self.ele_line()
		ans += '\n'.join(ele_line) + '\n'
		lattices = self.get_lattices(ele_line)
		ele_info = [self.bokas_elements[i] for i in self.eles]
		ans += '\n'
		for idx, i in enumerate(ele_info):
			if self.eles[idx] in self.transition_temperatures:
				lattice_change = self.transition_temperatures[self.eles[idx]][1]
				T_change = self.transition_temperatures[self.eles[idx]][2]
			else:
				lattice_change = ''
			for key, item in self.lattice_info.items():
				if lattices[idx] == item:
					line = f'FUNCTION GHSER{self.eles[idx].upper()} 298.15 {"+" if np.sign(i[key]) else ""}{i[key] * self.conversion}; 6000 N !'
				elif key == lattice_change:
					if self.T_flag:
						line = f'FUNCTION G{key}{self.eles[idx].upper()} 298.15 {"+" if np.sign(i[key]) else ""}{i[key] * self.conversion}*( 1- (T - 298.15)/({T_change} - 298.15)) ; 6000 N !'
					else:
						line = f'FUNCTION G{key}{self.eles[idx].upper()} 298.15 {"+" if np.sign(i[key]) else ""}{i[key] * self.conversion}; 6000 N !'
				else:
					line = f'FUNCTION G{key}{self.eles[idx].upper()} 298.15 {"+" if np.sign(i[key]) else ""}{i[key] * self.conversion}; 6000 N !'
				ans += line + '\n'

		ans += "$-------------------------------------------------------------------------------\n TYPE_DEFINITION % SEQ *!\n DEFINE_SYSTEM_DEFAULT ELEMENT 2 !\nDEFAULT_COMMAND DEF_SYS_ELEMENT VA /- !\n"
		ans += '\n'

		return ans, lattices

	def element_data_SGTE(self, ans):

		ans += self.topline() + '\n'
		ans += self.two_line() + '\n'
		ele_line = self.ele_line()
		ans += '\n'.join(ele_line) + '\n'
		lattices = self.get_lattices(ele_line)
		ans += '\n'
		for idx, i in enumerate(self.eles):
			for key, value in self.sgteda[i.upper()].items():
				ans += value

			ans += '\n'
		ans += "$-------------------------------------------------------------------------------\n TYPE_DEFINITION % SEQ *!\n DEFINE_SYSTEM_DEFAULT ELEMENT 2 !\nDEFAULT_COMMAND DEF_SYS_ELEMENT VA /- !\n"
		ans += '\n'

		return ans, lattices

	def ss_writer(self, ans, lattices):
		binaries = list(MultinaryCombinations.create_multinary(self.eles, no_comb=[2]).values())[0]
		omegas = {i: self.binary_data[i] for i in binaries}
		binaries = self.eles + binaries
		for key, item in self.lattice_info.items():
			phase_line = f"PHASE {item}  %  1  1.0  !\n"
			constituent_line = f"CONSTITUENT {item}  :{','.join([i.upper() for i in self.eles])} :  !"
			ans += phase_line + constituent_line + '\n'
			for i in binaries:
				if i in self.eles:
					parameter_line = f"PARAMETER G({item},{i.upper()};0)    "
					if lattices[self.eles.index(i)] == item:
						parameter_line += f"298.15 +GHSER{i.upper()}#;    6000 N!"
					else:
						parameter_line += f"298.15 +G{key}{i.upper()}#;    6000 N!"
					ans += parameter_line + '\n'
				else:
					if self.reg_flag:
						h_mix = omegas[i][key] * self.conversion
						i = i.split('-')
						parameter_line = f"PARAMETER L({item},{','.join([j.upper() for j in i])};0)    "
						parameter_line += f"298.15 {'+' if np.sign(h_mix) > 0 else ''}{h_mix};    6000 N!"
						ans += parameter_line + '\n'
					else:
						if isinstance(omegas[i][key], float):
							h_mix = omegas[i][key] * self.conversion
							i = i.split('-')
							parameter_line = f"PARAMETER L({item},{','.join([j.upper() for j in i])};0)    "
							parameter_line += f"298.15 {'+' if np.sign(h_mix) > 0 else ''}{h_mix};    6000 N!"
							ans += parameter_line + '\n'
						else:
							L0, L1 = omegas[i][key]
							L0, L1 = L0*self.conversion, L1*self.conversion
							i = i.split('-')
							parameter_line = f"PARAMETER L({item},{','.join([j.upper() for j in i])};0)    "
							parameter_line += f"298.15 {'+' if np.sign(L0) > 0 else ''}{L0};    6000 N!\n"
							parameter_line += f"PARAMETER L({item},{','.join([j.upper() for j in i])};1)    "
							parameter_line += f"298.15 {'+' if np.sign(L1) > 0 else ''}{L1};    6000 N!\n"
							ans += parameter_line + '\n'

			ans += '\n'

		return ans, binaries

	def liquid_writer(self, ans):
		binaries = list(MultinaryCombinations.create_multinary(self.eles, no_comb=[2]).values())[0]
		omegas = {i: self.binary_data[i] for i in binaries}
		binaries = self.eles + binaries
		key = 'LIQ'
		item = 'LIQUID'
		phase_line = f"PHASE {item}  %  1  1.0  !\n"
		constituent_line = f"CONSTITUENT {item}  :{','.join([i.upper() for i in self.eles])} :  !"
		ans += phase_line + constituent_line + '\n'
		for i in binaries:
			if i in self.eles:
				parameter_line = f"PARAMETER G({item},{i.upper()};0)    "
				parameter_line += f"{Element(i).melting_point} +G{key}{i.upper()}#;    6000 N!"
				ans += parameter_line + '\n'
			else:
				h_mix = omegas[i][key] * self.conversion
				i = i.split('-')
				parameter_line = f"PARAMETER L({item},{','.join([j.upper() for j in i])};0)    "
				parameter_line += f"298.15 {'+' if np.sign(h_mix) > 0 else ''}{h_mix};    6000 N!"
				ans += parameter_line + '\n'
			ans += '\n'

		return ans, binaries

	def im_writer(self, ans, binaries):
		entries = []
		for i in binaries:
			if i in self.im_list:
				entries.extend(self.im_list[i])

		for entry in entries:
			comp = entry.composition.as_dict()
			energy = entry.energy
			total_atoms = sum(list(comp.values()))
			phase_line = f"PHASE {entry.name.upper().replace(' ', '')}  %  {str(len(list(comp.keys())))} {'  '.join([str(i) for i in comp.values()])} ! \n"
			constituent_line = f"CONSTITUENT {entry.name.upper().replace(' ', '')}  :{' : '.join([i.upper() for i in comp.keys()])} :  ! \n"
			parameter_line = f"PARAMETER G({entry.name.upper().replace(' ', '')},{':'.join([i.upper() for i in comp.keys()])};0)  298.15    {energy * self.conversion} + {' +'.join([str(value) + '*' + 'GHSER' + key.upper() + '#' for key, value in comp.items()])};     6000 N !\n"
			# parameter_line = f"PARAMETER G({entry.name.upper().replace(' ', '')},{':'.join([i.upper() for i in comp.keys()])};0)  298.15    {energy * self.conversion};     6000 N !\n"
			ans += phase_line + constituent_line + parameter_line + "\n"
		return ans

	def break_long_lines(self, ans, width=113):
		new_lines = []
		for line in ans.splitlines():
			while len(line) > width:
				new_lines.append(line[:width])
				line = line[width:]
			new_lines.append(line)
		return '\n'.join(new_lines)

	def file_writer(self):
		ans = ''
		if self.SGTE:
			ans, lattices = self.element_data_SGTE(ans)
		else:
			ans, lattices = self.element_data(ans)

		# ans, binaries = self.liquid_writer(ans)
		ans, binaries = self.ss_writer(ans, lattices)

		self.ans = self.im_writer(ans, binaries)
		# self.ans = self.break_long_lines(ans)


	def file_saver(self):
		if self.reg:
			with open(f'../data/tdbs/{self.composition}_reg.tdb', 'w') as f:
				f.write(self.ans)
		else:
			with open(f'../data/tdbs/{self.composition}.tdb', 'w') as f:
				f.write(self.ans)