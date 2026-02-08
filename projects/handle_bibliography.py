import re
from bs4 import BeautifulSoup

# Chemins des fichiers
input_file = "_projects/detection_monture.md"  # Fichier texte à modifier
output_file = "_projects/detection_monture_corrected.md"  # Fichier de sortie
bibliography_file = r"_includes/bibliography.html"  # Fichier de bibliographie

# Charger le fichier de bibliographie
with open(bibliography_file, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# Créer un dictionnaire pour mapper les clés de citation aux numéros
citation_map = {}
for tr in soup.find_all("tr"):
    a_tag = tr.find("a", {"name": True})
    if a_tag:
        key = a_tag["name"]
        num = a_tag.text.strip("[]")
        citation_map[key] = num

# Lire le fichier texte
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Fonction pour remplacer les citations
def replace_citation(match):
    key = match.group(1)  # Récupérer la clé entre les accolades
    if key in citation_map:
        return f"[[{citation_map[key]}]]({key})"
    else:
        return match.group(0)  # Si la clé n'est pas trouvée, ne pas modifier

# Remplacer les commandes \cite{} par [[numero]](cle_bibliography)
pattern = r"\\cite\{([^}]+)\}"
corrected_content = re.sub(pattern, replace_citation, content)

# Écrire le résultat dans un nouveau fichier
with open(output_file, "w", encoding="utf-8") as f:
    f.write(corrected_content)

print(f"Le fichier corrigé a été enregistré sous : {output_file}")