Dieses Repository enthält eine vollständig lokal lauffähige Edge-AI-Anwendung zur Sprachverarbeitung auf Basis eines fein-getunten WAV2VEC2-Modells und eines kompakten Custom-Classifiers.
Alle Modelle befinden sich im Ordner models/ und werden über Git LFS bereitgestellt.
Nach dem Klonen des Repositories sollten daher zunächst die folgenden Befehle ausgeführt werden:

git lfs install
git lfs pull


Dadurch werden die großen Modelldateien (wav2vec.onnx_12 und classifier.onnx_12) heruntergeladen.
Die Anwendung läuft vollständig offline, nutzt ONNX Runtime für die Inferenz und misst während der Laufzeit CPU- und RAM-Verbrauch, Latenz sowie Konfidenzwerte der Klassifikation.

Das Projekt kann unmittelbar reproduziert werden:
Die Datei main.py greift für die Inferenz auf die Modelle wav2vec.onnx_12 und classifier.onnx_12 zu.
Mit den Modellen classifier_model_12.pt, classifier_model_13.pt und classifier_model_35.pt können die trainierten Klassifikatoren mithilfe des Skripts onnx_export.py erneut in das ONNX-Format exportiert und anschließend ebenfalls von der main.py verwendet werden.

Sollte das große WAV2VEC-Modell nicht vorhanden sein, kann es durch folgenden Befehl aus der Torchaudio-Pipeline automatisch neu erzeugt werden:

python onnx_export.py --export-wav2vec


Nach dem Start der Anwendung genügt ein Klick auf „Dauerhaft lauschen“:
Das Wake-Word lautet „MARVIN“. Wird es erkannt, kann eines der definierten Kommando-Wörter gesprochen werden, worauf die Anwendung die entsprechende Aktion ausführt.

Im linken Bereich des Interfaces werden das Monitoring (CPU, RAM, Latenz) und die geladenen Modelle angezeigt.
Falls das Endgerät offline ist, wird dies dort ebenfalls markiert – das Modell funktioniert dennoch uneingeschränkt weiter.
