"""Publication-exportable plot of the existing BINARY validation sweep only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
rows=json.loads((root/'reference/binary_gain_validation.json').read_text())
gains=sorted({r['phase_gain'] for r in rows});alphas=sorted({r['alpha'] for r in rows})
fig,ax=plt.subplots(figsize=(7.5,4.4),layout='constrained')
for alpha in alphas:
    values=[100*next(r['validation_correct']/r['validation_size'] for r in rows if r['phase_gain']==gain and r['alpha']==alpha) for gain in gains]
    ax.plot(range(len(gains)),values,marker='o',label=f'Ridge alpha = {alpha:g}',linewidth=1.6)
selected=next(r for r in rows if r['phase_gain']==.25 and r['alpha']==1000)
y=100*selected['validation_correct']/selected['validation_size']
ax.scatter([gains.index(.25)],[y],s=110,facecolors='none',edgecolors='black',zorder=5)
ax.annotate('Selected on this validation split',(gains.index(.25),y),xytext=(1.45,64.0),
            arrowprops={'arrowstyle':'->','color':'black'},fontsize=9)
ax.set_xticks(range(len(gains)),[f'{v:g}' for v in gains]);ax.set_ylim(58,64.5)
ax.set_xlabel('Phase gain (discrete values tested)');ax.set_ylabel('Validation accuracy (%)')
ax.set_title('Binary CIFAR-10: fixed two-bin readout, 8,000 / 2,000 split',fontsize=11)
ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=8)
fig.savefig(root/'results/binary_gain_validation.png',dpi=170)
fig.savefig(root/'results/binary_gain_validation.svg')
fig.savefig(root/'results/binary_gain_validation.pdf')
plt.close(fig)
