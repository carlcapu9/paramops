# CurveForge — guida in italiano

CurveForge è un add-on per Blender che distribuisce e adatta oggetti lungo le
curve, come RailClone in 3ds Max. Si lavora con i **nodi**: in un editor di nodi
(l'equivalente dello Style Editor di RailClone) si costruisce uno **stile**
collegando nodi Spline e Segment agli operatori e a un Linear Generator, che
dispone i segmenti lungo le curve. Il risultato è un oggetto mesh "live" che si
aggiorna mentre modifichi le curve, gli oggetti sorgente o i nodi.

![Muro con finestre, pilastri agli angoli e porta, con il suo stile](docs/overview.jpg)

* Blender **4.2 LTS → 5.x** (provato su 4.2.23 e 5.0.1).
* Il risultato è un normale oggetto mesh: si renderizza ovunque, mantiene
  materiali, UV e attributi e accetta altri modificatori. *Convert to Mesh*
  crea una copia indipendente dall'add-on.
* Documentazione completa (in inglese): [README.md](README.md).

## Installazione

1. Scarica `dist/curveforge-1.0.0.zip` (oppure crealo con `python build.py`).
2. In Blender: *Edit → Preferences → Get Extensions → ⌄ → Install from Disk…*
   e scegli lo zip (oppure trascina lo zip nella finestra di Blender).
3. Gli strumenti sono nella sidebar della vista 3D (`N`) → scheda
   **CurveForge**, nel menu *Add → New Curve Scatter / CurveForge Example* e
   nell'editor di nodi (tipo di editor **CurveForge Style**).

## Primo utilizzo

1. Modella il segmento lungo **+X** con **+Z verso l'alto**. La curva passa
   per l'origine dell'oggetto: Y = 0 è sulla curva, Z = 0 è l'altezza della
   curva.
2. Disegna una curva (Bezier, Poly o NURBS, aperta o chiusa, una o più spline).
3. Seleziona il segmento, poi la curva (attiva) e premi **New Curve Scatter**
   nella scheda CurveForge. Ottieni un oggetto CurveForge con il suo stile:
   `Spline → Linear Generator ← Segment (Default)`.
4. Premi **Edit Style** per aprire lo stile in un editor di nodi. Aggiungi nodi
   con `Shift+A` (Input, Generator, Operator, Number, Layout) e collegali.
5. Tutto è live: sposta i punti della curva, modifica gli oggetti, cambia i
   valori dei nodi.

*Add → CurveForge Example* crea sei stili pronti da studiare: muro con
finestre, recinzione, ringhiera, cordolo, sequenza, bordura da giardino.

![I sei esempi renderizzati con Cycles](docs/examples.jpg)

## Come funziona uno stile

```
Spline ─────────────────────────────┐
Segment ── Operatori (facoltativi) ─┼──▶ Linear Generator ──▶ oggetto CurveForge
Value / Segment Info / Math ... ────┘     (Start, End, Corner, Evenly, Default, Marker)
```

Il **Linear Generator** divide ogni spline in *sezioni* (agli angoli, ai marker
e ai cambi di Segment ID) e le riempie:

| Ingresso | Dove viene messo |
|----------|------------------|
| **Start / End** | All'inizio e alla fine delle spline aperte. |
| **Corner** | Sugli angoli (punti che girano più dell'angolo impostato, oppure tutti i punti). |
| **Evenly** | A distanza regolare (o in numero fisso) dentro ogni sezione. |
| **Marker** | In punti scelti, a distanze (`2.5, 50%, -1`) o ogni *Step*. |
| **Default** | Ripetuto per riempire il resto. **Adaptive**: stirato perché ne entri un numero intero; altrimenti misura reale con allineamento e avanzo (taglio, scala o vuoto). |

Gli operatori stanno tra i segmenti e il generatore e vengono valutati **per
ogni segmento posizionato**, con variabili come l'indice, la distanza lungo la
spline, la sezione, il Segment ID o la pendenza. I nodi numerici si possono
collegare a qualsiasi presa numerica (spaziatura, distanza evenly, offset,
trasformazioni, condizioni…), quindi ogni parametro può variare lungo la curva.

![Lo stile della ringhiera: Compose, Randomize e un montante su Start, End ed Evenly](docs/railing_style.jpg)

## I nodi

### Input

* **Spline** — un oggetto curva. *Splines* sceglie alcune spline (`0, 2, 4-6`,
  vuoto = tutte). *Reverse* inverte la direzione, *Resolution* (passi per tratto
  curvo, 0 = automatico), *Twist* (*Z Up* tiene i segmenti in piano, *Minimum*
  per percorsi 3D), *Use Tilt*. Più nodi Spline possono entrare nello stesso
  generatore.
* **Segment** — la geometria da ripetere: un **Object** (mesh, curva, testo…,
  con i suoi modificatori) oppure una **Collection** usata a caso (*Random*, con
  i pesi elencati nella sidebar del nodo), in sequenza (*Sequence*) o tutta
  insieme come un solo segmento (*Combine*). Per ogni segmento:
  * *Bend / Vertical / Slice / Instance*: Default (usa il generatore), On, Off.
    *Bend* deforma il segmento lungo la curva con giunti a mitra; *Vertical* lo
    tiene verticale sulle pendenze; *Slice* permette di tagliarlo alle estremità
    e sulle aree di ritaglio.
  * *Adaptive*: se il segmento può essere stirato.
  * Allineamento X (Auto, Pivot, Left, Center, Right), Y e Z, più *Offset*.
  * *Rotation*, *Scale*, *Mirror* X/Y/Z, *Custom Size* (lunghezza occupata
    sulla curva), *Padding* (spazio prima e dopo), *Corner Orientation* per i
    segmenti sugli angoli.
* **Empty Segment** — uno spazio vuoto lungo *Length*.

### Generator

* **Linear Generator** — dispone i segmenti lungo le spline collegate.
  * Ingressi: *Spline*, *Start*, *End*, *Corner*, *Evenly*, *Default*, *Marker*
    e le prese numeriche *Spacing*, *Evenly Distance*, *Evenly Count*,
    *Clip Start*, *Clip End*, *Offset Y*, *Offset Z*.
  * Limiti: ritaglio delle spline (distanza o %), *Extend* oltre le estremità
    con valori negativi, *Clipping Area* (curva chiusa: tiene dentro o fuori, i
    segmenti vengono tagliati sul bordo).
  * Segmenti Default: *Adaptive* con *Fit* Nearest / Shrink / Stretch / Count,
    oppure misura reale con *Align* e *Remainder*.
  * Angoli: *Sharp* (soglia d'angolo), *All Points* o *None*; *Split at
    Corners*; *Split at ID Changes*.
  * Evenly: a distanza o in numero, *Offset*, misurato per sezione o su tutta la
    spline.
  * Marker: *Points* (`1, 3, 5-7, all`), *Distances* (`2.5, 50%, -1`, i negativi
    contano dalla fine) o *Repeat* (ogni *Step* a partire da *Offset*).
  * Deformazione predefinita (*Bend*, *Vertical*, *Slice*, *Instancing*), *UV*
    (mantieni, oppure U lungo la curva), *Weld*, *Max Segments*, *Seed*.

### Operatori

* **Compose** — unisce gli ingressi in un unico segmento, in fila o
  sovrapposti, con uno spazio opzionale.
* **Sequence** — usa gli ingressi a turno; ogni presa ha un *Count*. Ricomincia
  a ogni spline, sezione, ingresso del generatore o mai.
* **Randomize** — sceglie un ingresso a caso; ogni presa ha un *Weight* (peso).
* **Conditional** — sceglie *True* o *False* confrontando una variabile con un
  valore (=, ≠, <, ≤, >, ≥, tra, pari, dispari, ogni N) oppure con la presa
  *Condition* (es. un nodo Expression).
* **Selector** — usa l'ingresso indicato da una variabile (di default il
  Segment ID) o dalla presa *Index*.
* **Mirror** — specchia su X/Y/Z sempre, a segmenti alterni o a caso.
* **Transform** — prese *Offset*, *Rotation*, *Scale* (pilotabili per segmento)
  più spostamento, rotazione (anche a scatti) e scala casuali.
* **Material** — sostituisce tutti i materiali o uno slot: il primo, a caso, in
  sequenza o per indice (fino a 8 materiali).
* **UV Transform** — sposta, scala e ruota le UV, anche a caso.

### Numeri

* **Value**, **Integer**, **Combine XYZ**.
* **Segment Info** — una variabile del segmento in corso (tabella sotto).
* **Math** — operazioni, confronti, and/or/not, pari, dispari, ogni N.
* **Random Value** — tra *Min* e *Max*, per segmento, per spline o per indice.
* **Expression** — una formula con le variabili, gli ingressi `a b c d` e
  funzioni come `min max clamp lerp sin cos floor round`. Esempio:
  `index % 3 == 1 and from_end > 0` (una finestra ogni tre pannelli, mai
  sull'ultimo).

| Variabile | Significato |
|-----------|-------------|
| `index` | Indice del segmento tra quelli dello stesso ingresso sulla spline |
| `global_index` | Indice tra tutti gli ingressi della spline |
| `input` | Ingresso: 0 Default, 1 Start, 2 End, 3 Corner, 4 Evenly, 5 Marker |
| `section`, `section_index`, `section_count`, `from_end` | Numero di sezione, indice dentro la sezione, segmenti nella sezione, segmenti rimasti fino alla fine |
| `distance`, `distance_pct` | Distanza dall'inizio della spline (lunghezza oppure 0–1) |
| `spline_length`, `section_length` | Lunghezze |
| `spline`, `spline_material` | Numero della spline e suo indice materiale |
| `segment_id` | Segment ID del tratto di curva |
| `marker` | Numero del marker |
| `corner_angle` | Angolo di svolta dell'angolo in gradi |
| `slope` | Pendenza della curva in gradi |
| `x`, `y`, `z` | Posizione nel mondo dell'inizio del segmento |
| `random` | Valore casuale 0–1 diverso per ogni segmento |

## Segment ID

Sono l'equivalente dei Material ID delle spline in RailClone: un numero per ogni
tratto di curva (tra due punti). In Edit Mode sulla curva seleziona punti
consecutivi e usa il pannello **Segment IDs** (*Assign*, *Select*, *Deselect*,
*Clear*). Gli ID si vedono nel viewport e seguono i loro punti quando modifichi
la curva. Si usano con un **Selector** (legge `segment_id` di default), con un
Conditional o con qualsiasi nodo numerico; i cambi di ID dividono le sezioni.

![Segment ID che scelgono siepe, staccionata o muretto](docs/segment_ids.jpg)

## Altro

* **Parametri esportati** — attiva *Show in Panel* su un nodo (sidebar
  dell'editor di nodi → CurveForge): le sue impostazioni compaiono nella sidebar
  della vista 3D sotto *Parameters*.
* **Instancing** — i segmenti rigidi (Bend spento, non tagliati) possono essere
  istanze dei loro oggetti (*Instancing* del generatore o *Instance* del
  segmento): scene più leggere.
* **Display** — *Boxes* mostra solo i bounding box; il pannello mostra segmenti,
  facce e tempo di calcolo; *Max Segments* evita errori costosi.
* **Animazione** — i valori dei nodi si possono animare; *Update on Frame
  Change* ricostruisce a ogni frame.

## Da RailClone a CurveForge

| RailClone | CurveForge |
|-----------|------------|
| Style Editor | editor di nodi *CurveForge Style* |
| Generatore Linear 1S (Start, End, Corner, Evenly, Default, Marker) | Linear Generator |
| Spline / Segment / segmento vuoto | Spline / Segment / Empty Segment |
| Compose, Sequence, Randomize, Conditional, Selector, Mirror, Transform, Material, UVW Xform | stessi nomi |
| Constant, Arithmetic, Random, Segment parameters, Expression | Value / Integer, Math, Random Value, Segment Info, Expression |
| Material ID delle spline | Segment ID |
| Clipping area, Extend, Clip spline | Clipping Area, Extend, Clip Start / End |
| Bend, Vertical, Slice, Adaptive, allineamento, padding | uguali, per generatore e per segmento |
| Instancing, Weld, UV in coordinate reali | Instancing, Weld, UV *Along Curve* |
| Parametri esportati | *Show in Panel* |

Non inclusi: il generatore Array 2S (griglie su due spline), le macro (gruppi
di nodi) e le luci come segmenti.
