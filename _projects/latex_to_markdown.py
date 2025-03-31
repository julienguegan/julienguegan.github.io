#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour convertir la syntaxe LaTeX en Markdown.
Peut être exécuté plusieurs fois sur différents fichiers.
"""

import os
import re
import sys
import argparse


def convert_latex_to_markdown(latex_content):
    """Convertit le contenu LaTeX en Markdown."""
    
    # Conversion des titres
    markdown_content = re.sub(r'\\section{(.*?)}', r'# \1', latex_content)
    markdown_content = re.sub(r'\\subsection{(.*?)}', r'## \1', markdown_content)
    markdown_content = re.sub(r'\\subsubsection{(.*?)}', r'### \1', markdown_content)
    
    # Conversion du formatage de texte
    markdown_content = re.sub(r'\\textit{(.*?)}', r'*\1*', markdown_content)
    markdown_content = re.sub(r'\\textbf{(.*?)}', r'**\1**', markdown_content)
    markdown_content = re.sub(r'\\emph{(.*?)}', r'*\1*', markdown_content)
    
    # Conversion des références
    markdown_content = re.sub(r'\\ref{(.*?)}', r'[\1]', markdown_content)
    markdown_content = re.sub(r'\\cite{(.*?)}', r'[\1]', markdown_content)
    
    # Conversion des listes
    # Supprimer les options de itemize/enumerate
    markdown_content = re.sub(r'\\begin{itemize}(\[.*?\])?', r'', markdown_content)
    markdown_content = re.sub(r'\\begin{enumerate}(\[.*?\])?', r'', markdown_content)
    markdown_content = re.sub(r'\\end{itemize}', r'', markdown_content)
    markdown_content = re.sub(r'\\end{enumerate}', r'', markdown_content)
    
    # Convertir les items en tirets pour les listes
    markdown_content = re.sub(r'\\item\s+(.*?)(?=\\item|\\end{|$)', r'- \1\n', markdown_content)
    
    # Conversion des tableaux (simplifiée)
    # Note: La conversion complète des tableaux LaTeX est complexe et peut nécessiter plus de logique
    
    # Conversion des équations (simplifiée)
    markdown_content = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', markdown_content)
    markdown_content = re.sub(r'\$(.*?)\$', r'$\1$', markdown_content)
    
    # Nettoyage final
    # Supprimer les lignes vides consécutives
    markdown_content = re.sub(r'\n\s*\n\s*\n', r'\n\n', markdown_content)
    
    return markdown_content


def process_file(input_file, output_file=None):
    """Traite un fichier LaTeX et le convertit en Markdown."""
    
    # Si aucun fichier de sortie n'est spécifié, créer un nom par défaut
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_markdown.md"
    
    try:
        # Lire le fichier d'entrée
        with open(input_file, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        
        # Convertir le contenu
        markdown_content = convert_latex_to_markdown(latex_content)
        
        # Écrire dans le fichier de sortie
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Conversion réussie : {input_file} -> {output_file}")
        return True
    
    except Exception as e:
        print(f"Erreur lors de la conversion de {input_file} : {str(e)}")
        return False


def main():
    """Fonction principale du script."""
    
    parser = argparse.ArgumentParser(description='Convertit des fichiers LaTeX en Markdown.')
    parser.add_argument('files', nargs='+', help='Fichiers LaTeX à convertir')
    parser.add_argument('-o', '--output', help='Dossier de sortie pour les fichiers convertis')
    parser.add_argument('-s', '--suffix', default='_markdown', help='Suffixe à ajouter aux noms des fichiers de sortie')
    
    args = parser.parse_args()
    
    success_count = 0
    
    for input_file in args.files:
        if not os.path.exists(input_file):
            print(f"Le fichier {input_file} n'existe pas.")
            continue
        
        if args.output:
            # Créer le dossier de sortie s'il n'existe pas
            os.makedirs(args.output, exist_ok=True)
            
            # Construire le chemin de sortie
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(args.output, f"{base_name}{args.suffix}.md")
        else:
            # Utiliser le même dossier que le fichier d'entrée
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}{args.suffix}.md"
        
        if process_file(input_file, output_file):
            success_count += 1
    
    print(f"\nRésumé : {success_count}/{len(args.files)} fichiers convertis avec succès.")


if __name__ == "__main__":
    main()