# Melody Maker - Genetic Algorithm for Musical Composition

Een genetisch algoritme systeem voor het genereren van muziekcomposities met geavanceerde heuristieken en automatische fitness evaluatie.

## 6 Gebruik en Deployment

### 6.1 Installatie

#### Vereisten
- Python 3.12+
- pip (Python package manager)
- Audio playback capability (voor het afspelen van gegenereerde muziek)

#### Installatiestappen

1. **Clone het project**
```bash
git clone https://github.com/IAkdim/TINLML2425-Opdracht2.git
cd TINLML2425-Opdracht2
```

2. **Installeer dependencies met pip**
```bash
pip install -r requirements.txt
```

De belangrijkste dependencies zijn:
- `matplotlib` - Voor het genereren van visualisaties en plots
- Standaard Python libraries: `random`, `copy`, `math`, `subprocess`

3. **Verificeer installatie**
```bash
python melody_maker_phase2.py
```

### 6.2 Gebruik

Het systeem biedt verschillende uitvoeringsmodi:

#### Command-line Opties

**Hoofdscript uitvoeren:**
```bash
python melody_maker_phase2.py
```

**Beschikbare modi:**
1. **Full Test Suite** - Uitgebreide research modus met 5 testscenario's
2. **Simple Run** - Eenvoudige compositie generatie

#### Invoerparameters

**Algemene GA Parameters:**
- `population_size`: Populatiegrootte (aanbevolen: 20-50)
- `mutation_rate`: Mutatiesnelheid (0.0-1.0, standaard: 0.12)
- `crossover_rate`: Crossover snelheid (0.0-1.0, standaard: 0.8)
- `crossover_type`: Type crossover (`single` of `uniform`)
- `mutation_strategy`: Mutatiestrategie (`pitch`, `rhythm`, of `both`)
- `generations`: Aantal generaties (aanbevolen: 30-100)

#### Output Bestanden

**Muziekbestanden:**
- `song.wav` - Finale compositie (Simple Run modus)
- `phase2_final_best.wav` - Beste overall compositie

**Visualisaties (alleen Full Test Suite):**
- `fitness_comparison_[timestamp].png` - Fitness vergelijking
- `diversity_comparison_[timestamp].png` - Populatie diversiteit
- `combined_results_dashboard_[timestamp].png` - Uitgebreid dashboard
