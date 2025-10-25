import torch
import torchaudio
from feature_classifier_model_12 import custom_net

model_path = r'C:\GitHub\EdgeAI\classifier_model_12.pt' #nimmt Modell 12

try:
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    feature_model = bundle.get_model()
    feature_model.eval()
    dummy_audio = torch.randn(1, 16000)  # 1 Sekunde bei 16kHz
    torch.onnx.export(feature_model, dummy_audio, "wav2vec.onnx_12")#speichert wav2vec_modell_12
    print('Feature_Modell geladen')

    model_as_dict = torch.load(model_path, weights_only=False)
    num_classes = model_as_dict['num_classes']

    classifier = custom_net(num_classes=num_classes)
    classifier.load_state_dict(model_as_dict['model_state_dict'])
    classifier.eval()
    dummy_feat = torch.randn(1,768)
    torch.onnx.export(classifier,dummy_feat, 'classifier.onnx_12')#speichert als Edge Modell 12

except Exception as e:
    print(f'Fehler: {e}')




