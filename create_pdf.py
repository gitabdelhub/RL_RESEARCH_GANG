#!/usr/bin/env python3
"""
Script simple pour convertir le rapport en PDF
"""

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.units import inch
    
    # Créer le PDF
    doc = SimpleDocTemplate("rapport_exercice2.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Lire le fichier texte
    with open('rapport_exercice2.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ajouter le contenu
    lines = content.split('\n')
    for line in lines:
        if line.strip():
            if line.startswith('#'):
                text = line.replace('#', '').strip()
                p = Paragraph(text, styles['Heading1'])
                story.append(p)
            elif line.startswith('##'):
                text = line.replace('##', '').strip()
                p = Paragraph(text, styles['Heading2'])
                story.append(p)
            elif line.startswith('###'):
                text = line.replace('###', '').strip()
                p = Paragraph(text, styles['Heading3'])
                story.append(p)
            elif line.strip() == '---':
                pass  # Ignorer les séparateurs
            else:
                p = Paragraph(line, styles['Normal'])
                story.append(p)
        else:
            story.append(Paragraph("<br/>", styles['Normal']))
    
    # Sauvegarder le PDF
    doc.build(story)
    print("PDF cree avec succes: rapport_exercice2.pdf")
    
except ImportError:
    print("La bibliotheque reportlab n'est pas installee.")
    print("Pour l'installer: pip install reportlab")
    print("Ou convertissez manuellement le fichier rapport_exercice2.txt en PDF.")
