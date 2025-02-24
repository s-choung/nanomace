import matplotlib.font_manager as fm
fss = 12  # Font size for legend
fs = 12  # Font size for ticks
fsss = 8  # Font size for annotate
fsl = 24  # Font size for labels
font_properties_label = fm.FontProperties(family='Arial', size=fs)
font_properties_tick = fm.FontProperties(family='Arial', size=fss)
font_properties_annotate = fm.FontProperties(family='Arial', size=fsss)

def subscript(text):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return text.translate(subscripts)
# Data
edge_color = 'black'
bar_width = 1  # Adjust the bar width as needed

ce_weight=140.116
o_weight=15.999 
abogadro_num= 6.02*1E+23

colors =['#840032','#e59500','#002642','gray']
bulk_threshold=4
markers=['o','s','^', 'v', 'D', '*']

