import subprocess
import base64
import csv
import io

# Get base64 of image
with open('aim2_method_v2_expanded.png', 'rb') as f:
    b64_data = base64.b64encode(f.read()).decode('utf-8')

# Run tesseract
tsv_out = subprocess.check_output(['tesseract', 'aim2_method_v2_expanded.png', 'stdout', 'tsv'], stderr=subprocess.DEVNULL).decode('utf-8')

texts = []
reader = csv.DictReader(io.StringIO(tsv_out), delimiter='\t')
for row in reader:
    try:
        conf = int(row['conf'])
        if conf > 30 and row['text'].strip():
            # Add some padding/offset for text baseline
            x = int(row['left'])
            y = int(row['top']) + int(row['height']) * 0.8
            text = row['text'].strip()
            # filter out simple artifacts
            if len(text) < 2 and not text.isalnum():
                continue
            if text in ['EEE', 'nn', 'a=']:
                continue
            
            fontsize = max(10, int(row['height']) * 0.7)
            texts.append(f'<text x="{x}" y="{y}" style="font-size:{fontsize}px;font-family:sans-serif;fill:black">{text}</text>')
    except ValueError:
        pass

svg_template = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="640" height="640" viewBox="0 0 640 640" version="1.1" 
     xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd">
    <sodipodi:namedview pagecolor="#ffffff" bordercolor="#000000" borderopacity="0.25" inkscape:showpageshadow="2" inkscape:pageopacity="0.0" inkscape:pagecheckerboard="0" inkscape:deskcolor="#d1d1d1" inkscape:document-units="px" inkscape:zoom="1.0" inkscape:cx="320" inkscape:cy="320" inkscape:window-width="1440" inkscape:window-height="900" inkscape:window-x="0" inkscape:window-y="0" inkscape:window-maximized="0" />
    <g id="Reference_Image_Layer" inkscape:label="Reference Image" inkscape:groupmode="layer" style="opacity:0.5">
        <image width="640" height="640" xlink:href="data:image/png;base64,{b64_data}"/>
    </g>
    <g id="Editable_Text_Layer" inkscape:label="Editable Text" inkscape:groupmode="layer">
        {chr(10).join(texts)}
    </g>
</svg>
"""

with open('aim2_method_v2_editable.svg', 'w') as f:
    f.write(svg_template)

print("SVG generated successfully.")
