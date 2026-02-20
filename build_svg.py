import xml.etree.ElementTree as ET

def build_svg():
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
    ET.register_namespace("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")
    
    tree = ET.parse('aim2_method_v2.svg')
    root = tree.getroot()
    
    # Update dimensions
    root.set('viewBox', '0 0 1000 850')
    if 'width' in root.attrib:
        del root.attrib['width']
    if 'height' in root.attrib:
        del root.attrib['height']
    
    layer = root[2]
    
    # Cleanup previous generated elements (those without an 'id')
    to_keep = []
    for child in layer:
        if "id" in child.attrib:
            to_keep.append(child)
    layer.clear()
    layer.extend(to_keep)
    
    # Helper to add elements easily
    def add_rect(x, y, w, h, fill, stroke="none", rx=0):
        # Escaping hex colors isn't strict in attributes, but we pass them directly
        rect = ET.SubElement(layer, "rect")
        rect.attrib.update({
            "x": str(x), "y": str(y), "width": str(w), "height": str(h),
            "fill": fill, "stroke": stroke, "rx": str(rx)
        })
        return rect
        
    def add_text(x, y, text, size=16, weight="normal", fill="black", anchor="start"):
        t = ET.SubElement(layer, "text")
        t.attrib.update({
            "x": str(x), "y": str(y), "fill": fill,
            "font-family": "sans-serif", "font-size": f"{size}px", "font-weight": weight,
            "text-anchor": anchor
        })
        t.text = text
        return t
        
    def add_line(x1, y1, x2, y2, stroke="black", width=2, marker_end=False):
        style = f"stroke:{stroke};stroke-width:{width}"
        if marker_end:
            style += ";marker-end:url(#marker43)" # reusing existing marker!
        line = ET.SubElement(layer, "line")
        line.attrib.update({
            "x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2),
            "style": style
        })
        return line
        
    def add_polyline(points_str, stroke="black", width=2, marker_end=False):
        style = f"fill:none;stroke:{stroke};stroke-width:{width}"
        if marker_end:
            style += ";marker-end:url(#marker43)"
        poly = ET.SubElement(layer, "polyline")
        poly.attrib.update({
            "points": points_str,
            "style": style
        })
        return poly
    
    # Helper to draw a fish
    def draw_fish(cx, cy, scale=1.0, colors=None):
        if colors is None:
            colors = {"body": "#9e8e6e", "head": "#6d6fa2", "dorsal": "#b08d6d", "tail": "#7ab08d", "bottom": "#b06d7a"}
        
        # Tail
        tail = ET.SubElement(layer, "polygon")
        tail.attrib.update({
            "points": f"{cx-40*scale},{cy} {cx-80*scale},{cy-30*scale} {cx-80*scale},{cy+30*scale}",
            "fill": colors["tail"], "stroke": "black", "stroke-width": "1"
        })
        
        # Dorsal fin
        dorsal = ET.SubElement(layer, "polygon")
        dorsal.attrib.update({
            "points": f"{cx-20*scale},{cy-15*scale} {cx+10*scale},{cy-40*scale} {cx+20*scale},{cy-15*scale}",
            "fill": colors["dorsal"], "stroke": "black", "stroke-width": "1"
        })
        
        # Bottom fin
        bottom = ET.SubElement(layer, "polygon")
        bottom.attrib.update({
            "points": f"{cx-10*scale},{cy+15*scale} {cx+10*scale},{cy+35*scale} {cx+20*scale},{cy+15*scale}",
            "fill": colors["bottom"], "stroke": "black", "stroke-width": "1"
        })
        
        # Body
        body = ET.SubElement(layer, "ellipse")
        body.attrib.update({
            "cx": str(cx), "cy": str(cy), "rx": str(50*scale), "ry": str(25*scale),
            "fill": colors["body"], "stroke": "black", "stroke-width": "1"
        })
        
        # Head (overlay on right part of body)
        head = ET.SubElement(layer, "path")
        head.attrib.update({
            "d": f"M {cx+20*scale} {cy-22*scale} A 50 25 0 0 1 {cx+50*scale} {cy} A 50 25 0 0 1 {cx+20*scale} {cy+22*scale} Z",
            "fill": colors["head"], "stroke": "black", "stroke-width": "1"
        })
        
        # Eye
        eye = ET.SubElement(layer, "circle")
        eye.attrib.update({
            "cx": str(cx+35*scale), "cy": str(cy-5*scale), "r": str(3*scale),
            "fill": "black"
        })

    # Part B
    add_line(0, 267, 800, 267, width=1)
    add_text(18, 285, "B: Multi-Recolor (Clustering/Quantization)", size=20, weight="bold")
    
    # Connector from A
    add_polyline("675,225 675,287 150,287 150,330", marker_end=True)
    
    # Gray boxes
    add_rect(128, 296, 352, 222, fill="#dbdbdb", stroke="black", rx=10)
    add_rect(141, 322, 327, 183, fill="#ececec", rx=10)
    add_rect(153, 347, 302, 146, fill="#f3f3f3", rx=10)
    
    add_text(305, 316, "Recolor Process", size=14, anchor="middle", weight="bold")
    add_text(305, 340, "Clustering/Quantization", size=14, anchor="middle")
    
    # Swatches Row 1
    swatch_y1 = 370
    swatch_x1 = 168
    orig_colors1 = ["#1a1a1a", "#4d433b", "#82786b", "#b3a595", "#d9cbba"]
    for i, c in enumerate(orig_colors1):
        add_rect(swatch_x1 + i*18, swatch_y1, 18, 25, fill=c, stroke="black")
    
    add_line(275, 381, 335, 381, marker_end=True)
    
    swatch_x_q1 = 356
    quant_colors1 = ["#1c1c1a", "#5a6b78", "#78856b", "#957361", "#ccbc99"]
    for i, c in enumerate(quant_colors1):
        add_rect(swatch_x_q1 + i*18, swatch_y1, 18, 25, fill=c, stroke="black")
        
    add_text(swatch_x1 + 45, swatch_y1 + 40, "Original", size=12, anchor="middle")
    add_text(swatch_x_q1 + 45, swatch_y1 + 40, "Quantized", size=12, anchor="middle")
    
    # Description Text for Part B
    add_text(160, 480, "1. Extracts original texture maps from 3D models.", size=12, fill="#333333")
    add_text(160, 495, "2. Applies k-means clustering to group similar colors.", size=12, fill="#333333")
    add_text(160, 510, "3. Quantizes the texture maps to a specified palette size.", size=12, fill="#333333")
    
    # Swatches Row 2
    swatch_y2 = 430
    swatch_x2 = 168
    orig_colors2 = ["#3a2a2a", "#5d534b", "#92887b", "#c3b5a5", "#e9dbca"] # Random slight variation
    for i, c in enumerate(orig_colors2):
        add_rect(swatch_x2 + i*18, swatch_y2, 18, 25, fill=c, stroke="black")
    
    add_line(275, 441, 335, 441, marker_end=True)
    
    swatch_x_q2 = 356
    # Same quant colors for example
    for i, c in enumerate(quant_colors1):
        add_rect(swatch_x_q2 + i*18, swatch_y2, 18, 25, fill=c, stroke="black")

    # Output Arrow
    add_line(481, 406, 540, 406, width=3, marker_end=True)
    
    # Quantized Texture Fish
    base_fish_colors = {"body": "#dbdbdb", "head": "#dbdbdb", "dorsal": "#dbdbdb", "tail": "#dbdbdb", "bottom": "#dbdbdb"}
    # Let's just use generic colors for the first one if we wanted to
    patterned_colors = {"body": quant_colors1[4], "head": quant_colors1[1], "dorsal": quant_colors1[2], "tail": quant_colors1[0], "bottom": quant_colors1[3]}
    draw_fish(625, 400, scale=1.2, colors=patterned_colors)
    add_text(625, 460, "Quantized Texture", size=14, anchor="middle")

    # Part C
    add_line(0, 537, 800, 537, width=1)
    add_text(18, 555, "C: Segmentation (Mesh Region Selection)", size=20, weight="bold")
    
    # Connector from B
    add_polyline("625,475 625,510 150,510 150,570", marker_end=True)
    
    # Gray boxes
    add_rect(128, 570, 352, 211, fill="#dbdbdb", stroke="black", rx=10)
    add_rect(141, 597, 327, 171, fill="#ececec", rx=10)
    
    add_text(305, 587, "Segmentation", size=14, anchor="middle", weight="bold")
    add_text(305, 615, "Mesh Region Selection", size=14, anchor="middle")
    
    # Wireframe Fish
    wireframe_colors = {"body": "#d66ea3", "head": "#4a90e2", "dorsal": "#8fc269", "tail": "#f5db6e", "bottom": "#f5db6e"}
    draw_fish(305, 680, scale=1.2, colors=wireframe_colors)
    
    # Add wireframe grid lines over the fish manually as a simple representation
    for i in range(-50, 60, 15):
        add_line(305+i, 680-25, 305+i, 680+25, stroke="white", width=1)
        
    # Description Text for Part C
    add_text(150, 740, "1. Define region boundary using Closed Curve markup.", size=12, fill="#333333")
    add_text(150, 755, "2. InterDeCA selects all vertices within the bounded area.", size=12, fill="#333333")
    add_text(150, 770, "3. Export the highlighted region as a separate 3D model.", size=12, fill="#333333")

    # Output Arrow
    add_line(481, 681, 550, 681, width=3, marker_end=True)
    
    # Segmented Mesh Image
    seg_colors = {"body": "#9e8e6e", "head": "#6d6fa2", "dorsal": "#b08d6d", "tail": "#7ab08d", "bottom": "#b06d7a"}
    draw_fish(650, 680, scale=1.4, colors=seg_colors)
    
    # Callout Labels
    add_text(600, 610, "Fins", size=12)
    add_line(610, 615, 630, 640, width=1)
    
    add_text(650, 605, "Body", size=12, anchor="middle")
    add_line(650, 610, 650, 650, width=1)
    
    add_text(710, 630, "Head", size=12)
    add_line(715, 635, 685, 670, width=1)
    
    add_text(630, 755, "Fins", size=12)
    add_line(640, 740, 660, 720, width=1)
    
    add_text(590, 740, "Tail", size=12)
    add_line(600, 730, 610, 700, width=1)
    
    add_text(650, 770, "Segmented Mesh", size=14, anchor="middle", weight="bold")
    
    # Save the updated SVG
    tree.write('aim2_method_v2.svg', encoding='utf-8', xml_declaration=True)

build_svg()
print("Done reconstructing SVG.")
