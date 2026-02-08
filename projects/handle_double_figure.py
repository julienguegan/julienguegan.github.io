import re

# Chemins des fichiers
input_file = "_projects/detection_monture.md" # Fichier Markdown à modifier
output_file = "_projects/detection_monture_corrected.md" # Fichier de sortie

# Lire le fichier Markdown
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Expression régulière pour capturer les blocs de figures LaTeX avec subfigure
figure_pattern = re.compile(
    r"\\begin\{figure\}\[H\][\s\S]*?"  # \begin{figure}[H]
    r"\\centering[\s\S]*?"  # \centering
    r"(\\begin\{subfigure\}[\s\S]*?\\end\{subfigure\}[\s\S]*?)"  # Première subfigure
    r"(\\begin\{subfigure\}[\s\S]*?\\end\{subfigure\}[\s\S]*?)"  # Deuxième subfigure
    r"\\caption\{(.*?)\}[\s\S]*?"  # Légende principale de la figure
    r"\\label\{.*?\}[\s\S]*?"  # Étiquette de la figure (ignorée ici)
    r"\\end\{figure\}",  # \end{figure}
    re.DOTALL  # Permet de capturer des blocs multilignes
)

# Expression régulière pour extraire les informations de chaque subfigure
subfigure_pattern = re.compile(
    r"\\begin\{subfigure\}.*?\{.*?\}[\s\S]*?"  # \begin{subfigure}[t]{0.45\textwidth}
    r"\\centering[\s\S]*?"  # \centering
    r"\\includegraphics\[.*?\]\{(.*?)\}[\s\S]*?"  # \includegraphics[scale=0.5]{image.png}
    r"\\caption\{(.*?)\}[\s\S]*?"  # Légende de la subfigure
    r"\\end\{subfigure\}",  # \end{subfigure}
    re.DOTALL  # Permet de capturer des blocs multilignes
)

# Fonction pour remplacer les figures LaTeX avec subfigures par du HTML
def replace_figure_with_subfigures(match):
    subfigure1 = match.group(1)  # Première subfigure
    subfigure2 = match.group(2)  # Deuxième subfigure
    main_caption = match.group(3)  # Légende principale de la figure
    
    # Extraire les informations de chaque subfigure
    subfigure1_info = subfigure_pattern.search(subfigure1)
    subfigure2_info = subfigure_pattern.search(subfigure2)
    
    if not subfigure1_info or not subfigure2_info:
        return match.group(0)  # Retourner le texte original si les informations ne sont pas trouvées
    
    # Extraire les chemins d'image et les légendes
    image_path1, caption1 = subfigure1_info.groups()
    image_path2, caption2 = subfigure2_info.groups()
    
    # Convertir les chemins d'image en chemins relatifs pour HTML
    image_name1 = image_path1.split("/")[-1].replace(" ", "_")
    image_name2 = image_path2.split("/")[-1].replace(" ", "_")
    html_image_path1 = f"/assets/images/{image_name1}"
    html_image_path2 = f"/assets/images/{image_name2}"
    
    # Générer le code HTML pour les deux subfigures
    html_figure = (
        f'<div style="display: flex; justify-content: center; gap: 10%;">\n'
        f'  <div style="width: 40%;">\n'
        f'    <img src="{html_image_path1}" alt="Images" style="width: 100%;"/>\n'
        f'    <p align="center"><i>{caption1}</i></p>\n'
        f'  </div>\n'
        f'  <div style="width: 40%;">\n'
        f'    <img src="{html_image_path2}" alt="Masques" style="width: 100%;"/>\n'
        f'    <p align="center"><i>{caption2}</i></p>\n'
        f'  </div>\n'
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