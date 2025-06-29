# Melody Maker - Genetic Algorithm for Musical Composition

Een genetisch algoritme systeem voor het genereren van muziekcomposities met geavanceerde heuristieken en automatische fitness evaluatie.

## 6 Gebruik en Deployment

### 6.1 Installatie

#### Vereisten
- Python 3.12+
- uv (Python package manager)
- Audio playback capability (voor het afspelen van gegenereerde muziek)

#### Installatiestappen

1. **Clone het project**
```bash
git clone <repository-url>
cd music-maker
```

2. **Installeer dependencies met uv**
```bash
uv sync
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
3. **Interactive Mode** - Stap-voor-stap evolutie controle

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
- `test_X_[crossover]_[mutation]_best.wav` - Beste composities per test
- `phase2_final_best.wav` - Beste overall compositie

**Visualisaties (alleen Full Test Suite):**
- `fitness_comparison_[timestamp].png` - Fitness vergelijking
- `diversity_comparison_[timestamp].png` - Populatie diversiteit
- `combined_results_dashboard_[timestamp].png` - Uitgebreid dashboard

### 6.3 Voorbeeldworkflow

#### Scenario 1: Genereer een enkele compositie (`song.wav`)

```bash
# 1. Start het programma
python melody_maker_phase2.py

# 2. Kies Simple Run modus
Choose mode:
1. Full test suite (5 scenarios with plots)
2. Simple run (just evolve and generate song.wav)
Enter choice (1/2, default 1): 2

# 3. Configureer parameters (of gebruik defaults)
Population size (default 30): 40
Mutation rate 0.0-1.0 (default 0.12): 0.15
Crossover rate 0.0-1.0 (default 0.8): 0.8
Crossover type (single/uniform, default uniform): uniform
Mutation strategy (pitch/rhythm/both, default both): both
Number of generations (default 50): 75

# 4. Wacht op evolutie (automatisch)
# Het systeem toont voortgang elke 10 generaties

# 5. Resultaat: song.wav wordt gegenereerd
```

**Verwachte output:**
```
Evolution complete!
Final best fitness: 78.5/100
Generating final composition...
Final composition saved as: song.wav

Composition details:
  Melody track: 42 notes
  Bass track: 16 notes
  Final diversity: 15.32
```

#### Scenario 2: Uitgebreide research analyse

```bash
# 1. Start het programma
python melody_maker_phase2.py

# 2. Kies Full Test Suite
Enter choice (1/2, default 1): 1

# 3. Configureer research parameters
Population size (default 15): 15
Mutation rate 0.0-1.0 (default 0.12): 0.12
Crossover rate 0.0-1.0 (default 0.8): 0.8
Number of generations (default 10): 10

# 4. Systeem voert 5 tests uit:
# - Uniform + pitch
# - Uniform + rhythm  
# - Uniform + both
# - Single + both
# - Single + pitch

# 5. Resultaten worden gegenereerd
```

**Verwachte output:**
```
=== 5.3 RESULTATEN (SAMENVATTING) ===
Test                 Best Fitness    Gemiddelde Fitness   Diversiteit (laatste gen.)
--------------------------------------------------------------------------------
Uniform + pitch      89.7            71.2                 5.4
Uniform + rhythm     86.3            69.8                 6.1
Uniform + both       92.4            74.5                 5.9
Single + both        84.1            67.0                 7.2
Single + pitch       81.5            66.3                 6.7

=� Generated 3 visualization files:
  " fitness_comparison_20250629_143025.png
  " diversity_comparison_20250629_143025.png  
  " combined_results_dashboard_20250629_143025.png

<� Generated 5 audio files for best compositions
```

#### Scenario 3: Snelle test met specifieke parameters

Voor ontwikkeling of quick testing:

```bash
# Gebruik kleine parameters voor snelle resultaten
Population size: 10
Generations: 15
# Resultaat binnen 1-2 minuten
```

#### Scenario 4: Productie-kwaliteit compositie

Voor beste muziekkwaliteit:

```bash
# Gebruik grote parameters voor betere evolutie
Population size: 50
Generations: 100
Crossover type: uniform
Mutation strategy: both
# Resultaat binnen 10-15 minuten, hoge kwaliteit
```

### Troubleshooting

**Veel voorkomende problemen:**

1. **ModuleNotFoundError: matplotlib**
```bash
uv add matplotlib
```

2. **Geen audio output**
- Controleer of de `music/` directory bestaat
- Verificeer dat `music/muser.py` functioneert

3. **Lange uitvoertijd**
- Verlaag population size en/of generations
- Test eerst met kleine parameters

4. **Lage fitness scores**
- Verhoog aantal generaties
- Probeer verschillende mutation strategies
- Gebruik uniform crossover voor betere resultaten

### Performance Richtlijnen

**Geschatte uitvoertijden:**

| Parameters | Tijd | Kwaliteit |
|------------|------|-----------|
| Pop: 10, Gen: 15 | 1-2 min | Basis |
| Pop: 30, Gen: 50 | 5-8 min | Goed |
| Pop: 50, Gen: 100 | 15-20 min | Uitstekend |

**Aanbevolen configuraties:**

- **Quick test:** Population 10, Generations 15
- **Standard:** Population 30, Generations 50  
- **High quality:** Population 50, Generations 100
- **Research:** Population 15, Generations 10 (voor consistente vergelijking)
