import os

# Chemin du dossier contenant les images
dossier_images = "/home/julien/Documents/images"  # Remplacez par le chemin de votre dossier

# Liste des extensions d'images à traiter
extensions_images = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]

# Parcourir tous les fichiers du dossier
for fichier in os.listdir(dossier_images):
    # Vérifier si le fichier est une image
    if any(fichier.lower().endswith(ext) for ext in extensions_images):
        # Chemin complet du fichier actuel
        chemin_actuel = os.path.join(dossier_images, fichier)
        
        # Remplacer les espaces par des underscores
        nouveau_nom = fichier.replace(" ", "_")
        
        # Ajouter le suffixe "monture_" devant le nom
        nouveau_nom = "monture_" + nouveau_nom
        
        # Chemin complet du nouveau fichier
        nouveau_chemin = os.path.join(dossier_images, nouveau_nom)
        
        # Renommer le fichier
        os.rename(chemin_actuel, nouveau_chemin)
        print(f"Renommé : {fichier} -> {nouveau_nom}")

print("Renommage terminé.")