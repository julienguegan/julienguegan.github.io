import re

# Chemins des fichiers
input_file = "_projects/detection_monture.md" # Fichier Markdown à modifier
output_file = "_projects/detection_monture_corrected.md" # Fichier de sortie

# Lire le fichier Markdown
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Expression régulière pour capturer les blocs de figures LaTeX avec subfigure
figure_pattern = re.compile(
    r"\\begin\{figure\}\[H\]\s*"  # \begin{figure}[H]
    r"\\centering\s*"  # \centering
    r"(\\begin\{subfigure\}.*?\\end\{subfigure\}\s*)+"  # Une ou plusieurs subfigures
    r"\\caption\{(.*?)\}\s*"  # Légende principale de la figure
    r"\\label\{.*?\}\s*"  # Étiquette de la figure (ignorée ici)
    r"\\end\{figure\}",  # \end{figure}
    re.DOTALL  # Permet de capturer des blocs multilignes
)

# Expression régulière pour capturer chaque subfigure
subfigure_pattern = re.compile(
    r"\\begin\{subfigure\}\[.*?\]\{.*?\}\s*"  # \begin{subfigure}[t]{0.35\textwidth}
    r"\\centering\s*"  # \centering
    r"\\includegraphics\[.*?\]\{(.*?)\}\s*"  # \includegraphics[scale=0.5]{image.png}
    r"\\caption\{(.*?)\}\s*"  # Légende de la subfigure
    r"\\end\{subfigure\}",  # \end{subfigure}
    re.DOTALL  # Permet de capturer des blocs multilignes
)

# Fonction pour remplacer les figures LaTeX avec subfigures par du HTML
def replace_figure_with_subfigures(match):
    subfigures = match.group(1)  # Capture toutes les subfigures
    main_caption = match.group(2)  # Légende principale de la figure
    
    # Extraire chaque subfigure
    subfigure_matches = subfigure_pattern.findall(subfigures)
    
    # Générer le code HTML pour chaque subfigure
    html_subfigures = []
    for image_path, caption in subfigure_matches:
        # Convertir le chemin de l'image en chemin relatif pour HTML
        image_name = image_path.split("/")[-1].replace(" ", "_")  # Remplace les espaces par des underscores
        html_image_path = f"/assets/images/{image_name}"
        
        # Générer le code HTML pour une subfigure
        html_subfigure = (
            f'<div style="display: inline-block; width: {100 / len(subfigure_matches)}%; text-align: center;">\n'
            f'  <img src="{html_image_path}" style="width: 100%; max-width: 100%;"/>\n'
            f'  <p><i>{caption}</i></p>\n'
            f'</div>'
        )
        html_subfigures.append(html_subfigure)
    
    # Combiner les subfigures en une seule ligne
    html_figure = (
        f'<div style="display: flex; justify-content: space-between; align-items: center;">\n'
        f'  {"".join(html_subfigures)}\n'
        f'</div>\n'
        f'<p align="center">\n'
        f'  <i>{main_caption}</i>\n'
        f'</p>'
    )
    return html_figure

# Remplacer toutes les figures LaTeX avec subfigures par du HTML
corrected_content = figure_pattern.sub(replace_figure_with_subfigures, content)

# Écrire le résultat dans un nouveau fichier
with open(output_file, "w", encoding="utf-8") as f:
    f.write(corrected_content)

print(f"Le fichier corrigé a été enregistré sous : {output_file}")