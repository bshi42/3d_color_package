import xml.etree.ElementTree as ET
import random

def build_svg():
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    
    # Root element
    root = ET.Element("svg", {
        "viewBox": "0 0 1000 1200",
        "version": "1.1",
        "xmlns": "http://www.w3.org/2000/svg"
    })
    
    # Definitions for markers
    defs = ET.SubElement(root, "defs")
    # Arrow marker
    marker = ET.SubElement(defs, "marker", {
        "id": "arrow", "viewBox": "0 0 10 10", "refX": "9", "refY": "5",
        "markerWidth": "6", "markerHeight": "6", "orient": "auto-start-reverse"
    })
    ET.SubElement(marker, "path", {"d": "M 0 0 L 10 5 L 0 10 z", "fill": "black"})
    
    # Gradient for Step 3 Original Texture
    grad = ET.SubElement(defs, "linearGradient", {"id": "fish_grad", "x1": "0%", "y1": "0%", "x2": "100%", "y2": "0%"})
    ET.SubElement(grad, "stop", {"offset": "0%", "stop-color": "#8f7c6b"})
    ET.SubElement(grad, "stop", {"offset": "50%", "stop-color": "#b59f8c"})
    ET.SubElement(grad, "stop", {"offset": "100%", "stop-color": "#6e5e50"})
    
    group = ET.SubElement(root, "g")
    
    def add_rect(x, y, w, h, fill="white", stroke="black", rx=5, stroke_width="2"):
        r = ET.SubElement(group, "rect", {
            "x": str(x), "y": str(y), "width": str(w), "height": str(h),
            "fill": fill, "stroke": stroke, "rx": str(rx), "stroke-width": str(stroke_width)
        })
        return r
        
    def add_text(x, y, text, size=16, weight="normal", fill="black", anchor="start"):
        t = ET.SubElement(group, "text", {
            "x": str(x), "y": str(y), "fill": fill, "font-family": "sans-serif",
            "font-size": f"{size}px", "font-weight": weight, "text-anchor": anchor
        })
        t.text = text
        return t
        
    def add_line(x1, y1, x2, y2, width=3, marker=True):
        l = ET.SubElement(group, "line", {
            "x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2),
            "stroke": "black", "stroke-width": str(width)
        })
        if marker:
            l.attrib["marker-end"] = "url(#arrow)"
        return l
        
    def add_circle(cx, cy, r, fill, stroke="black"):
        ET.SubElement(group, "circle", {
            "cx": str(cx), "cy": str(cy), "r": str(r), "fill": fill, "stroke": stroke
        })

    def draw_fish(cx, cy, scale=1.0, colors=None, gradient=False):
        if colors is None:
            colors = {"body": "#dbdbdb", "head": "#dbdbdb", "dorsal": "#dbdbdb", "tail": "#dbdbdb"}
            
        body_fill = "url(#fish_grad)" if gradient else colors["body"]
        head_fill = "url(#fish_grad)" if gradient else colors["head"]
        dorsal_fill = "url(#fish_grad)" if gradient else colors["dorsal"]
        tail_fill = "url(#fish_grad)" if gradient else colors["tail"]

        ET.SubElement(group, "polygon", {
            "points": f"{cx-40*scale},{cy} {cx-80*scale},{cy-30*scale} {cx-80*scale},{cy+30*scale}",
            "fill": tail_fill, "stroke": "black", "stroke-width": "1"
        })
        ET.SubElement(group, "polygon", {
            "points": f"{cx-20*scale},{cy-15*scale} {cx+10*scale},{cy-40*scale} {cx+20*scale},{cy-15*scale}",
            "fill": dorsal_fill, "stroke": "black", "stroke-width": "1"
        })
        ET.SubElement(group, "ellipse", {
            "cx": str(cx), "cy": str(cy), "rx": str(50*scale), "ry": str(25*scale),
            "fill": body_fill, "stroke": "black", "stroke-width": "1"
        })
        ET.SubElement(group, "path", {
            "d": f"M {cx+20*scale} {cy-22*scale} A 50 25 0 0 1 {cx+50*scale} {cy} A 50 25 0 0 1 {cx+20*scale} {cy+22*scale} Z",
            "fill": head_fill, "stroke": "black", "stroke-width": "1"
        })
        ET.SubElement(group, "circle", {
            "cx": str(cx+35*scale), "cy": str(cy-5*scale), "r": str(3*scale),
            "fill": "black"
        })
        
    # Main Header
    add_text(500, 60, "Workflow: Multi-Recolor (Multi-texture Clustering)", size=32, weight="bold", anchor="middle")
    
    # ---------------------------------------------------------
    # Step 1: Extract Textures
    # ---------------------------------------------------------
    y1 = 150
    add_rect(50, y1, 900, 200, fill="#f9f9f9", stroke="#cccccc", stroke_width="1")
    add_text(70, y1+40, "Step 1: Extract Aligned Textures", size=24, weight="bold")
    add_text(70, y1+70, "Extracts aligned textures from the generated Atlas UV space for all specimens.", size=16, fill="#555")
    
    # Visuals
    draw_fish(200, y1+130, scale=1.5, gradient=True)
    add_text(200, y1+190, "3D Specimen", size=14, anchor="middle", weight="bold")
    
    add_line(300, y1+130, 450, y1+130)
    
    add_rect(500, y1+80, 150, 100, fill="url(#fish_grad)")
    add_text(575, y1+200, "UV Texture Flat Map", size=14, anchor="middle", weight="bold")

    # ---------------------------------------------------------
    # Step 2: K-Means Clustering
    # ---------------------------------------------------------
    y2 = 400
    add_rect(50, y2, 900, 200, fill="#f9f9f9", stroke="#cccccc", stroke_width="1")
    add_text(70, y2+40, "Step 2: K-Means Clustering for Shared Palette", size=24, weight="bold")
    add_text(70, y2+70, "Applies K-Means Clustering across all pooled textures to create a universal 8-color palette.", size=16, fill="#555")
    
    # Left side: randomly colored pixels
    random.seed(42)
    colors_pool = ["#4a3c31", "#635345", "#8c7a6b", "#b8a89a", "#3b453a", "#546b5a", "#7aa385", "#2a3b4c"]
    for i in range(10):
        for j in range(8):
            c = random.choice(colors_pool)
            add_rect(150 + i*12, y2+90 + j*12, 10, 10, fill=c, stroke="none", rx=0)
    add_text(205, y2+200, "Millions of Raw Pixels", size=14, anchor="middle", weight="bold")
            
    add_line(300, y2+130, 450, y2+130)
    
    # The Palette
    for i, c in enumerate(colors_pool):
        add_rect(500 + i*35, y2+110, 30, 40, fill=c)
    add_text(635, y2+180, "8 Consolidated Clusters", size=14, anchor="middle", weight="bold")

    # ---------------------------------------------------------
    # Step 3: Quantize Textures
    # ---------------------------------------------------------
    y3 = 650
    add_rect(50, y3, 900, 200, fill="#f9f9f9", stroke="#cccccc", stroke_width="1")
    add_text(70, y3+40, "Step 3: Quantize Individual Textures", size=24, weight="bold")
    add_text(70, y3+70, "Maps every pixel in each specimen to the nearest color in the shared palette.", size=16, fill="#555")
    
    draw_fish(200, y3+130, scale=1.5, gradient=True)
    add_text(200, y3+190, "Original Continuous Texture", size=14, anchor="middle", weight="bold")
    
    add_line(330, y3+130, 450, y3+130)
    
    q_colors = {"body": colors_pool[2], "head": colors_pool[7], "dorsal": colors_pool[6], "tail": colors_pool[4]}
    draw_fish(600, y3+130, scale=1.5, colors=q_colors)
    add_text(600, y3+190, "Quantized Texture (Discrete)", size=14, anchor="middle", weight="bold")

    # ---------------------------------------------------------
    # Step 4: Morphospace / PCA
    # ---------------------------------------------------------
    y4 = 900
    add_rect(50, y4, 900, 250, fill="#f9f9f9", stroke="#cccccc", stroke_width="1")
    add_text(70, y4+40, "Step 4: Morphospace / PCA Analysis", size=24, weight="bold")
    add_text(70, y4+70, "Performs PCA on quantized color patterns to construct a Morphospace for population comparison.", size=16, fill="#555")
    
    # Plot axes
    px, py = 500, y4+200
    add_line(px-150, py, px+150, py, marker=True) # X axis
    add_line(px, py+20, px, py-100, marker=True)  # Y axis
    add_text(px+160, py+5, "PC1", size=14, weight="bold")
    add_text(px-20, py-110, "PC2", size=14, weight="bold")
    
    # Scatter points
    dots = [(-100, -20), (-80, -40), (-120, -10), (50, -60), (70, -80), (40, -40), (10, -50), (-30, -70), (110, -20)]
    for dx, dy in dots:
        c = random.choice(colors_pool)
        add_circle(px+dx, py+dy, 6, fill=c)
        
    add_text(px, y4+230, "Specimen Scatter Plot in Morphospace", size=14, anchor="middle", weight="bold")

    # Write output
    with open('multi_recolor_workflow.svg', 'wb') as f:
        tree = ET.ElementTree(root)
        tree.write(f, encoding='utf-8', xml_declaration=True)

if __name__ == "__main__":
    build_svg()
    print("Successfully built multi_recolor_workflow.svg")
