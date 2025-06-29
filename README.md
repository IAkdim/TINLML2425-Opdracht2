# Music Maker - Genetic Algorithm Music Composer

Een intelligente muziekcompositor die genetische algoritmen gebruikt om automatisch melodieën en baslijnten te genereren.

## Installatie

### Vereisten
- Python 3.7+
- pip package manager

### Setup

1. **Clone of download het project:**
   ```bash
   cd music-maker
   ```

2. **Installeer dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Maak music directory aan (automatisch bij eerste run):**
   ```bash
   mkdir -p music
   ```

## Gebruik & Deployment

### Fase 1: Handmatige Fitness Evaluatie

Voor beginners en experimentele compositie met handmatige feedback:

```bash
python phase1.py
```

**Interactieve parameters:**
- Populatie grootte (standaard: 10)
- Mutatie kans (standaard: 0.15)
- Crossover kans (standaard: 0.7)
- Aantal generaties (standaard: 3)

### Fase 2: Geautomatiseerde Fitness met Heuristieken

Voor geavanceerde compositie met muzikale intelligentie:

```bash
python phase2.py
```

**Opties:**
1. **Enkele run** - Experimenteer met verschillende strategieën
2. **Vergelijking experimenten** - Test alle combinaties automatisch

## Command-line Parameters

### Fase 1 Parameters
- **Populatie grootte**: Aantal individuen per generatie (5-50)
- **Mutatie kans**: Waarschijnlijkheid van mutatie (0.05-0.3)
- **Crossover kans**: Waarschijnlijkheid van recombinatie (0.5-0.9)
- **Generaties**: Aantal evolutie cycli (1-20)

### Fase 2 Parameters
- **Crossover type**: 
  - `single` - Traditionele single-point crossover
  - `uniform` - Uniform crossover voor meer variatie
- **Mutatie strategie**:
  - `pitch` - Alleen toonhoogte mutatie
  - `rhythm` - Alleen ritme mutatie  
  - `both` - Combinatie van beide (aanbevolen)

## Voorbeeldworkflow: Snel een song.wav genereren

### Snelle Start (3 minuten)
```bash
# Automatische compositie met standaard instellingen
python phase2.py
# Kies: 1 (enkele run)
# Druk Enter voor alle standaard waarden
# Resultaat: beste_compositie_uniform_both.wav
```

### Geavanceerde Workflow (10 minuten)
```bash
# Fase 1: Experimenteer met handmatige feedback
python phase1.py
# Voer in: 15, 0.2, 0.8, 5
# Evalueer 3-5 composities handmatig
# Resultaat: finale_compositie.wav

# Fase 2: Verfijn met heuristieken
python phase2.py
# Kies: 2 (vergelijking experimenten)
# Bekijk alle strategieën automatisch
# Resultaat: 5 verschillende .wav bestanden
```

### Optimale Instellingen
Voor beste resultaten:
- **Populatie**: 15-20 individuen
- **Mutatie**: 0.12-0.15
- **Crossover**: 0.7-0.8
- **Strategie**: uniform + both
- **Generaties**: 8-12

## Output Files

### Gegenereerde Bestanden
- `song.wav` - Huidige compositie
- `finale_compositie.wav` - Beste resultaat Fase 1
- `beste_compositie_[strategie].wav` - Beste resultaat Fase 2
- `temp_evaluatie.wav` - Tijdelijke evaluatie bestanden

### Muzikale Eigenschappen
- **Formaat**: WAV audio
- **Tempo**: 130 BPM
- **Toonsoort**: C majeur
- **Structuur**: Melodie + Baslijn
- **Duur**: 15-60 seconden (afhankelijk van compositie)

## Configuratie

### Building Blocks
Het systeem gebruikt voorgedefinieerde muzikale patronen:
- **intro**: Openingsthema
- **verse**: Hoofdmelodie
- **chorus**: Refrein
- **bridge**: Overgang
- **outro**: Afsluiting
- **bass**: Baslijn patronen

### Fitness Criteria (Fase 2)
Het algoritme evalueert composities op:
1. **Lengte structuur** (25 punten) - Ideale maat indeling
2. **Akkoordprogressies** (25 punten) - Muzikale harmonie
3. **Melodische beweging** (20 punten) - Natuurlijke intervallen
4. **Ritme diversiteit** (15 punten) - Variatie in nootwaarden
5. **Muzikale afsluiting** (15 punten) - Correcte tonica eindigen

## Troubleshooting

### Veelvoorkomende Problemen
- **ModuleNotFoundError**: Voer `pip install -r requirements.txt` uit
- **Geen audio**: Controleer of `music/` directory bestaat
- **Lage fitness scores**: Probeer meer generaties of lagere mutatie
- **Te weinig variatie**: Verhoog mutatie kans of gebruik uniform crossover

### Performance Tips
- Gebruik populatie 10-20 voor snelle experimenten
- Gebruik populatie 20-50 voor kwaliteitsresultaten
- Meer generaties = betere convergentie maar langzamer
- Fase 2 is sneller dan Fase 1 (geen handmatige input)

## Technische Details

### Dependencies
- `tomita` - Muzieksynthese engine
- `numpy` - Numerieke berekeningen
- `matplotlib` - Optionele visualisatie
- `click` - Command-line interface

### Architectuur
- `muser.py` - Audio generatie engine
- `phase1.py` - Handmatige fitness GA
- `phase2.py` - Geautomatiseerde fitness GA
- `requirements.txt` - Python dependencies

Voor meer informatie over de algoritmen, zie de code commentaar in de fase bestanden.