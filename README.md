Edge-AI Sprachverarbeitung mit WAV2VEC2

Dieses Repository enthält eine vollständig lokal lauffähige Edge-AI-Anwendung zur Sprachverarbeitung auf Basis eines fein-getunten WAV2VEC2-Modells und eines kompakten Custom-Classifiers. Alle Modelle befinden sich im Ordner models/ und werden über Git LFS bereitgestellt.

Installation und Setup
Nach dem Klonen des Repositories müssen zunächst die großen Modelldateien über Git LFS heruntergeladen werden: 
git lfs install und git lfs pull. Dadurch werden die Modelldateien (wav2vec.onnx_12, classifier.onnx_12 und classifier_model_12.pt) heruntergeladen.
Bei Schwierigkeiten mit lokalen Abhängigkeiten sollte die requirements.txt verwendet werden, um die Umgebung einzurichten: 
pip install -r requirements.txt
WICHTIG: Vor dem ersten Start müssen die folgenden drei Pfade in main.py an die tatsächlichen Dateipfade in der lokalen Umgebung angepasst werden:
whole_model_path, onnx_feat_model_path und onnx_clf_model_path. Ersetze C:\GitHub\Test\ durch den Pfad zu deinem lokalen Repository-Ordner.

Verwendung
Die Anwendung läuft vollständig offline, nutzt ONNX Runtime für die Inferenz und misst während der Laufzeit CPU- und RAM-Verbrauch, Latenz sowie Konfidenzwerte der Klassifikation. 
Nach dem Start mit python main.py genügt ein Klick auf „Dauerhaft lauschen". Das Wake-Word lautet MARVIN. 
Nach Erkennung des Wake-Words kann eines der definierten Kommando-Wörter gesprochen werden (go, stop, on, off, up, down, left, right, yes, no), worauf die Anwendung die entsprechende Aktion ausführt.

Modell-Export und Reproduzierbarkeit
Das Projekt kann unmittelbar reproduziert werden. Die verwendeten Modelle sind wav2vec.onnx_12 (Feature-Extraktor), classifier.onnx_12 (Classifier) und classifier_model_12.pt (vollständiges PyTorch-Modell inkl. Label-Encoder). 
Mit den Modellen classifier_model_12.pt, classifier_model_13.pt und classifier_model_35.pt können die trainierten Klassifikatoren mithilfe des Skripts onnx_export.py erneut in das ONNX-Format exportiert werden. 
Sollte das große WAV2VEC-Modell nicht vorhanden sein, kann es durch python onnx_export.py --export-wav2vec aus der Torchaudio-Pipeline automatisch neu erzeugt werden.

Interface und Monitoring
Im Interface werden CPU-Last, RAM-Verbrauch, Latenz, die geladenen Modelle mit Größenangaben sowie der Internet-Status angezeigt. Falls das Endgerät offline ist, wird dies markiert – das Modell funktioniert dennoch uneingeschränkt weiter.
Die Anwendung basiert auf PyTorch, ONNX Runtime, Torchaudio, PyAudio, WebRTC VAD und Flet und ist vollständig offline-fähig.
