import pandas as pd
import numpy as np
from pycalphad import Database, equilibrium
from pycalphad import variables as v
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def composition_profile_per_temperature(composition, input_file_path, output_file_path, mols, temperature):

	eles = composition.split('-')
	df = Database(f"{input_file_path}/{composition}")
	comps = [i.upper() for i in eles] + ['VA']
	phases = list(df.phases.keys())
	df_predictions = []
	
	for mol in mols:
		mol_frac = dict(zip(comps[:-1], mol))
		feats = {v.X(i): mol_frac[i] for i in comps[:-2]}
		feats[v.T] = temperature
		feats[v.P] = 101325
		equi = equilibrium(df, comps, phases, feats)
		phase_info = np.squeeze(np.array(equi.Phase))
		phase_compositions = np.round(np.squeeze(np.array(equi.X)),2)
		phase_fractions = np.round(np.squeeze(np.array(equi.NP)),2)
		temp = list(mol) + list(phase_info) + list(phase_compositions) + list(phase_fractions)
		df_predictions.append(temp)
	
	df = pd.DataFrame(df_predictions)
	
	elements = composition.split('-')
	
	phase_name_cols = ['4', '5', '6', '7', '8']
	phase_comp_cols = ['9', '10', '11', '12', '13']
	phase_frac_cols = ['14', '15', '16', '17', '18']
	
	all_phases = []
	for col in phase_name_cols:
		all_phases.extend(df[col].dropna().unique())
	unique_phases = sorted(list(set(all_phases)))
	
	def clean_phase_name(p):
		return p.split('.')[-1].split('_')[0]
	
	cmap_phases = plt.get_cmap('Spectral')
	if len(unique_phases) > 1:
		phase_colors = {phase: cmap_phases(i / (len(unique_phases) - 1)) for i, phase in enumerate(unique_phases)}
	else:
		phase_colors = {unique_phases[0]: cmap_phases(0.5)}
	
	# Soothing colors for elements
	colors_elements = ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b']
	element_colors = {elem: colors_elements[i] for i, elem in enumerate(elements)}
	
	fig, ax = plt.subplots(figsize=(7.58, 3.58))
	N_alloys = len(df)
	
	y_positions = np.arange(N_alloys) * 0.4  # Smaller gap
	bar_height = 0.15  # Thinner bars
	
	y_labels = []
	y_ticks = []
	
	for i in range(N_alloys):
		lbl = f"Nb$_{{{df.loc[i, '0']:.2f}}}$Ti$_{{{df.loc[i, '1']:.2f}}}$V$_{{{df.loc[i, '2']:.2f}}}$Zr$_{{{df.loc[i, '3']:.2f}}}$"
		y_labels.append(lbl)
		y_ticks.append(y_positions[i])
		
		left_phase = 0.0
		left_elem = 0.0
		
		y_phase = y_positions[i] + bar_height / 2 + 0.005
		y_elem = y_positions[i] - bar_height / 2 - 0.005
		
		for j, p_col in enumerate(phase_name_cols):
			p = df.loc[i, p_col]
			if pd.notna(p):
				p_frac = df.loc[i, phase_frac_cols[j]]
				
				# Draw thick line between phases (before plotting the next phase if it's not the first)
				if left_phase > 0:
					ax.plot([left_phase, left_phase], [y_elem - bar_height / 2, y_phase + bar_height / 2],
					        color='black', linewidth=2, zorder=5)
				
				# Phase bar (Top)
				ax.barh(y_phase, p_frac, left=left_phase, height=bar_height,
				        color=phase_colors[p], edgecolor='black', linewidth=0.5, alpha=0.75)
				
				comp_str = df.loc[i, phase_comp_cols[j]]
				if pd.notna(comp_str) and isinstance(comp_str, str):
					comp_vals = np.fromstring(comp_str.replace('[', '').replace(']', ''), sep=' ')
					for k, elem in enumerate(elements):
						e_frac_in_phase = comp_vals[k]
						e_frac_total = e_frac_in_phase * p_frac
						
						if e_frac_total > 0:
							ax.barh(y_elem, e_frac_total, left=left_elem, height=bar_height,
							        color=element_colors[elem], edgecolor='black', linewidth=0.5, alpha=0.75)
							
							perc = int(round(e_frac_in_phase * 100))
							if e_frac_total > 0.03 and perc > 0:
								ax.text(left_elem + e_frac_total / 2, y_elem, f"{perc}",
								        ha='center', va='center', fontsize=8, color='black')
							
							left_elem += e_frac_total
				left_phase += p_frac
		
		# Draw thick line at the ends of the bars for consistency
		ax.plot([0, 0], [y_elem - bar_height / 2, y_phase + bar_height / 2], color='black', linewidth=2, zorder=5)
		ax.plot([1, 1], [y_elem - bar_height / 2, y_phase + bar_height / 2], color='black', linewidth=2, zorder=5)
	
	ax.set_yticks(y_ticks)
	ax.set_yticklabels(y_labels)
	ax.set_xlabel("Phase Fractions")
	ax.set_xlim(0, 1)

	legend_elements_phase = []
	for p in unique_phases:
		clean_p = clean_phase_name(p)
		legend_elements_phase.append(Patch(facecolor=phase_colors[p], edgecolor='black', label=clean_p, linewidth=0.5))
	
	legend_elements_elem = [Patch(facecolor=element_colors[e], edgecolor='black', label=e, linewidth=0.5) for e in elements]
	
	leg1 = ax.legend(handles=legend_elements_phase, title="Phases", bbox_to_anchor=(1.02, 1), loc='upper left',
	                 frameon=False)
	ax.add_artist(leg1)
	ax.legend(handles=legend_elements_elem, title="Elements", bbox_to_anchor=(1.02, 0.4), loc='upper left', frameon=False)
	
	plt.tight_layout()
	plt.savefig(f'{output_file_path}{composition}_{temperature}K.png', dpi=300)