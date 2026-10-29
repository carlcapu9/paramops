"""Italian translation of the interface (used when Blender's language is Italian).

Blender looks strings up by ``(context, message)``: property names, panel titles and
labels use the default context ``"*"``, operator names/buttons use ``"Operator"``.
Every message below is registered in both contexts.
"""

import bpy

IT = {
    # panels, menus, headings
    "Stress Sculpt": "Stress Sculpt",
    "Setup": "Impostazione",
    "Design Domain": "Dominio di progetto",
    "Pick a closed mesh as domain,": "Scegli una mesh chiusa come dominio,",
    "or start from a demo scene.": "oppure parti da una scena demo.",
    "Demo Scene": "Scena demo",
    "Demo": "Demo",
    "Forces & Zones": "Forze e zone",
    "Add at least one support": "Aggiungi almeno un appoggio",
    "Add at least one load": "Aggiungi almeno un carico",
    "Active Object": "Oggetto attivo",
    "This mesh is the design domain": "Questa mesh è il dominio di progetto",
    "Use as Domain": "Usa come dominio",
    "Force (N)": "Forza (N)",
    "The arrow (local +Z) is the force direction": "La freccia (+Z locale) è la direzione della forza",
    "Fixed Axes": "Assi bloccati",
    "Region": "Regione",
    "Shape": "Forma",
    "Size": "Dimensione",
    "The mesh interior is the zone": "L'interno della mesh è la zona",
    "Analyze": "Analizza",
    "Analyze Stress": "Analizza sollecitazioni",
    "Last result: ": "Ultimo risultato: ",
    "Max stress": "Tensione massima",
    "Max displacement": "Spostamento massimo",
    "Compliance": "Cedevolezza",
    "Evolve": "Evolvi",
    "Evolve Shape": "Fai evolvere la forma",
    "Growth Rules": "Regole di crescita",
    "Symmetry": "Simmetria",
    "'Hollow Below' should be lower than 'Thicken Above'": "'Svuota sotto' deve essere minore di 'Ispessisci sopra'",
    "Tension Lines": "Linee di tensione",
    "Branching": "Ramificazione",
    "Thickness": "Spessore",
    "Min": "Min",
    "Max": "Max",
    "Trace Tension Lines": "Traccia linee di tensione",
    "Sculpture": "Scultura",
    "Traces the lines first if needed": "Se serve, traccia prima le linee",
    "Sculpt from Lines": "Scolpisci dalle linee",
    "Relief": "Rilievo",
    "Thicken Surface": "Ispessisci superficie",
    "Variants": "Varianti",
    "Kind": "Tipo",
    "Restore Settings": "Ripristina impostazioni",
    "Compare": "Confronta",
    "Solver": "Risolutore",
    "Clear Results": "Cancella risultati",
    "Cancel": "Annulla",
    "Working": "Calcolo in corso",
    "Add Zone": "Aggiungi zona",
    "Support": "Appoggio",
    "Load": "Carico",
    "Zone": "Zona",
    # operators (names as shown in menus / search)
    "Use Active as Domain": "Usa l'attivo come dominio",
    "Set Domain": "Imposta dominio",
    "Add Support": "Aggiungi appoggio",
    "Add Load": "Aggiungi carico",
    "Set Role": "Imposta ruolo",
    "Select by Role": "Seleziona per ruolo",
    "Trace Lines": "Traccia linee",
    "Sculpt": "Scolpisci",
    "Save Variant": "Salva variante",
    "Remove Variant": "Rimuovi variante",
    "Move Variant": "Sposta variante",
    "Restore Variant Settings": "Ripristina impostazioni variante",
    "Compare Variants": "Confronta varianti",
    "Clear Comparison": "Cancella confronto",
    "Cancel Job": "Annulla calcolo",
    # roles
    "Role": "Ruolo",
    "None": "Nessuno",
    "Keep Solid": "Mantieni pieno",
    "Keep Void": "Mantieni vuoto",
    "Thicken": "Ispessisci",
    "Hollow": "Svuota",
    "Branch": "Ramifica",
    # object settings
    "Enabled": "Attivo",
    "Force": "Forza",
    "Spread Radius": "Raggio di distribuzione",
    "Strength": "Intensità",
    # scene settings
    "Domain": "Dominio",
    "Resolution": "Risoluzione",
    "Voxelize": "Voxelizza",
    "Mesh Interior": "Interno della mesh",
    "Bounding Box": "Ingombro",
    "Material": "Materiale",
    "Custom": "Personalizzato",
    "PLA (3D print)": "PLA (stampa 3D)",
    "Steel": "Acciaio",
    "Aluminium": "Alluminio",
    "Concrete": "Calcestruzzo",
    "Wood": "Legno",
    "Bone": "Osso",
    "Young's Modulus (GPa)": "Modulo di Young (GPa)",
    "Poisson Ratio": "Coefficiente di Poisson",
    "Multigrid": "Multigriglia",
    "Jacobi": "Jacobi",
    "Tolerance": "Tolleranza",
    "Max Solver Iterations": "Iterazioni massime del risolutore",
    "Show Domain as Wire": "Mostra il dominio in wireframe",
    "Show": "Mostra",
    "Von Mises": "Von Mises",
    "Max Principal (tension)": "Principale massima (trazione)",
    "Min Principal (compression)": "Principale minima (compressione)",
    "Displacement": "Spostamento",
    "Colors": "Colori",
    "Heat": "Calore",
    "Cool / Warm": "Freddo / Caldo",
    "Viridis": "Viridis",
    "Clip Percentile": "Percentile di saturazione",
    "Color Original Shape": "Colora la forma originale",
    "Method": "Metodo",
    "Optimize": "Ottimizza",
    "Penalty": "Penalizzazione",
    "Smoothness": "Morbidezza",
    "Move Limit": "Limite di passo",
    "Iterations": "Iterazioni",
    "Convergence": "Convergenza",
    "Thicken Above": "Ispessisci sopra",
    "Hollow Below": "Svuota sotto",
    "Rate": "Velocità",
    "Steer to Material Amount": "Tendi alla quantità di materiale",
    "Surface Level": "Livello della superficie",
    "Smoothing": "Levigatura",
    "Live Preview": "Anteprima dal vivo",
    "Stress From": "Tensioni da",
    "Full Domain": "Dominio pieno",
    "Evolved Shape": "Forma evoluta",
    "Lines": "Linee",
    "Tension + Compression": "Trazione + compressione",
    "Tension": "Trazione",
    "Compression": "Compressione",
    "Seeds": "Semi",
    "Seed at Loads & Supports": "Semi su carichi e appoggi",
    "Step": "Passo",
    "Separation": "Distanza",
    "Min Stress": "Tensione minima",
    "Branch Levels": "Livelli di ramificazione",
    "Branch Spacing": "Spaziatura dei rami",
    "Branch Threshold": "Soglia di ramificazione",
    "Random Seed": "Seme casuale",
    "Min Radius": "Raggio minimo",
    "Max Radius": "Raggio massimo",
    "Branch Taper": "Rastremazione dei rami",
    "Fuse": "Fondi",
    "Lines + Evolved": "Linee + forma evoluta",
    "Detail": "Dettaglio",
    "Blend": "Raccordo",
    "Carve Low Stress": "Scava dove la tensione è bassa",
    "Neutral Stress": "Tensione neutra",
    "Contrast": "Contrasto",
    "Subdivisions": "Suddivisioni",
    "Spacing": "Spaziatura",
    "Advanced": "Avanzate",
    "Notes": "Note",
    # demo presets
    "Bridge": "Ponte",
    "Cantilever": "Mensola",
    "Tower": "Torre",
    "Chair": "Sedia",
}

translations_dict = {
    "it_IT": {
        (context, message): translation
        for message, translation in IT.items()
        for context in ("*", "Operator")
    }
}


def register():
    try:
        bpy.app.translations.unregister(__package__)
    except (ValueError, RuntimeError):
        pass
    bpy.app.translations.register(__package__, translations_dict)


def unregister():
    try:
        bpy.app.translations.unregister(__package__)
    except (ValueError, RuntimeError):
        pass
