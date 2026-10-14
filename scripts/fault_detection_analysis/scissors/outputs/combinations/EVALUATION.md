# Evaluatie van extra parametercombinaties

De uitbreiding heeft 68 afzonderlijke residukenmerken en alle 2.278 paren
geanalyseerd. De zeven bestaande klasse-indelingen en drie bestaande
knipselecties blijven behouden. Iedere geselecteerde knip is een afzonderlijke
observatie en complete runs blijven buiten training. De gewone LDA-beslisregel
met gelijke priors en automatische shrinkage is ongewijzigd.

## Healthy (0) versus alle after-filing-iteraties (1,2,3,4,6,8)

| Selectie | Beste oude telling TP/FN/TN/FP | Beste uitgebreide telling TP/FN/TN/FP |
|---|---|---|
| Alle knips | 33 / 3 / 6 / 0 | 34 / 2 / 6 / 0 |
| Alleen knip 1 | 12 / 0 / 2 / 0 | 12 / 0 / 2 / 0 |
| Knips 2 en 3 afzonderlijk | 24 / 0 / 4 / 0 | 24 / 0 / 4 / 0 |

De winst bij alle knips is één extra correct herkende after-filing-knip.
Er zijn drie kandidaten met deze beste tellingen:

- M3 closing second Fourier amplitude + M3 opening positive integral.
- M3 opening mean + M0–M3 active-cycle waveform correlation.
- M3 opening positive integral + M3 active-cycle mean.

Bij knip 1 blijven onder andere M0 opening absolute maximum en range foutloos.
Bij knips 2 en 3 delen meerdere paren de foutloze tellingen; de eerder gekozen
M0 + M3 opening mean blijft dus een gelijkwaardige kandidaat op basis van deze tellingen.

Het concrete voorbeeld M0 closing RMS + M3 closing absolute maximum scoort
26 TP, 10 FN, 5 TN en 1 FP bij alle knips. Dat paar is geen verbetering.

## Extra controle van de parameterkeuze zelf

De beste kandidaat is gekozen na het vergelijken van 2.346 kandidaten op
dezelfde dataset. Zijn gewone per-candidate cross-validation-resultaat blijft
daarom verkennend. Een aanvullende nested analyse koos de kandidaat per
outer fold alleen met de overgebleven trainingsruns:

| Selectie | TP | FN | TN | FP |
|---|---:|---:|---:|---:|
| Alle knips | 29 | 7 | 2 | 4 |
| Alleen knip 1 | 12 | 0 | 2 | 0 |
| Knips 2 en 3 | 21 | 3 | 2 | 2 |

Dit zijn resultaten van een wisselende selectieprocedure, niet van één vast
parameterpaar. Ze laten zien dat de keuze bij alle knips en knips 2,3 instabiel
is. Het mooie resultaat van de best gevonden vaste kandidaat mag daarom niet
als onafhankelijk bevestigde toekomstige detectieprestatie worden gepresenteerd.

Deze nested controle heeft een duidelijke beperking: er zijn maar twee healthy
runs. Na het buitenhouden van één healthy run is er nog maar één beschikbaar
voor training. Die kan niet nogmaals als inner testset dienen zonder de
healthy trainingsklasse te verliezen. Zulke inner folds worden overgeslagen,
waardoor de inner selectie in die gevallen geen healthy testobservaties heeft.
Dit is per fold expliciet vastgelegd. Ook het foutloze eerste-knip-resultaat
bevat slechts twee healthy testobservaties, van twee runs op dit instrument.

## Exacte severity-labels, alle knips

De onderstaande vertegenwoordigers maximaliseren balanced accuracy voor de
exacte drie labels; aantallen blijven de primaire rapportage.

| Intermediate-iteraties | Healthy correct | Intermediate correct | Blunt correct | Fault als healthy |
|---|---|---|---|---:|
| 1,2,3,4,6 | 6/6 | 20/30 | 6/6 | 4 |
| 1,4,6 | 6/6 | 15/18 | 6/6 | 2 |
| Alleen 4 | 6/6 | 6/6 | 6/6 | 0 |

De eerste rij gebruikt M3 closing signed maximum + M3 opening first Fourier
amplitude; de tweede M3 closing signed maximum + M3 opening mean.
De derde gebruikt één parameter: M3 closing range.
Exacte severity-classificatie en het vermijden van gemiste fault alarms
zijn verschillende doelen. `top_models.csv` en de aparte figuren geven
beide rangschikkingen weer. De interpretatie blijft beperkt tot iteration-defined
groepen; er is geen onafhankelijke kwantitatieve maat voor mechanische botheid.

## Verificatie en reproduceerbaarheid

Alle 44.928 vergelijkbare bestaande standaard-threshold-testvoorspellingen
van 117 oorspronkelijke modellen zijn gereproduceerd. Vier oude controlemodellen
met samplegemiddelden vallen buiten de 68 primaire kenmerken.
Daarnaast zijn 2.520 nieuwe testbeslissingen rechtstreeks vergeleken met
scikit-learn; ze zijn identiek. Maximale coefficient- en interceptverschillen
zijn ongeveer 1e-11 en 5e-11.

De PNG/SVG-grenzen zijn full-data fits voor illustratie. De aparte confusion
matrices en tellingen komen uit withheld predictions; ze zijn geen trainingtellingen.
De dataset en DT-voorspellingen zijn niet aangepast.
