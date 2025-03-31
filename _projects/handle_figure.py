import re

def process_figure_references(markdown_content):
    # Identifier les figures et leur attribuer un numéro
    figure_pattern = r'<figure[^>]*id="(fig:[^"]+)"[^>]*>'
    figures = re.findall(figure_pattern, markdown_content)
    
    # Créer un dictionnaire de correspondance id -> numéro
    figure_numbers = {fig_id: i+1 for i, fig_id in enumerate(figures)}
    
    # Remplacer les références dans le texte
    for fig_id, number in figure_numbers.items():
        # Remplacer les références comme [fig:id] par [Figure X](#fig:id)
        markdown_content = re.sub(
            r'\[' + fig_id + r'\]',
            r'[Figure ' + str(number) + r'](#' + fig_id + r')',
            markdown_content
        )
        
        # Remplacer les références comme [Figure X](#fig:id) par [Figure Y](#fig:id)
        markdown_content = re.sub(
            r'\[Figure \d+\]\(#' + fig_id + r'\)',
            r'[Figure ' + str(number) + r'](#' + fig_id + r')',
            markdown_content
        )
    
    return markdown_content

# Chemins des fichiers
input_file = "_projects/detection_monture.md"  # Fichier Markdown à modifier
output_file = "_projects/detection_monture_corrected.md"  # Fichier de sortie

# Lire le fichier Markdown
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Expression régulière pour capturer les blocs de figures LaTeX
figure_pattern = re.compile(
    r"\\begin\{figure\}\[H\]\s*"  # \begin{figure}[H]
    r"\\centering\s*"  # \centering
    r"\\includegraphics\[.*?\]\{(.*?)\}\s*"  # \includegraphics[scale=0.5]{image.png}
    r"\\caption\{(.*?)\}\s*"  # \caption{Description de la figure}
    r"\\label\{.*?\}\s*"  # \label{fig:example}
    r"\\end\{figure\}",  # \end{figure}
    re.DOTALL  # Permet de capturer des blocs multilignes
)

# Fonction pour remplacer les figures LaTeX par du HTML
def replace_figure(match):
    image_path = match.group(1)  # Chemin de l'image (ex: "Figures/semantic segmentation.png")
    caption = match.group(2)  # Légende de la figure (ex: "Compromis évolutifs")
    
    # Convertir le chemin de l'image en chemin relatif pour HTML
    # Exemple : "Figures/semantic segmentation.png" -> "/assets/images/semantic_segmentation.png"
    image_name = image_path.split("/")[-1].replace(" ", "_")  # Remplace les espaces par des underscores
    html_image_path = f"/assets/images/{image_name}"
    
    # Générer le code HTML
    html_figure = (
        f'<p align="center">\n'
        f'  <img src="{html_image_path}" width="60%"/>\n'
        f'</p>\n'
        f'<p align="center">\n'
        f'  <i>{caption}</i>\n'
        f'</p>'
    )
    return html_figure

# Remplacer toutes les figures LaTeX par du HTML
corrected_content = figure_pattern.sub(replace_figure, content)

# Écrire le résultat dans un nouveau fichier
with open(output_file, "w", encoding="utf-8") as f:
    f.write(corrected_content)

print(f"Le fichier corrigé a été enregistré sous : {output_file}")