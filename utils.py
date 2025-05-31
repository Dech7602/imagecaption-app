
import torch
import pickle
from torchvision.models.detection import maskrcnn_resnet50_fpn
from positional_transformer import ImageCaptioningTransformer  # your model
from vocabulary import Vocabulary  # your vocab class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    # Caption model
    caption_model = ImageCaptioningTransformer(vocab_size=5000)
    caption_model.load_state_dict(torch.load("caption_model.pth", map_location=device))
    caption_model.to(device)
    caption_model.eval()

    # Segmentation model
    seg_model = maskrcnn_resnet50_fpn(pretrained=False)
    seg_model.load_state_dict(torch.load("segment_model.pth", map_location=device))
    seg_model.to(device)
    seg_model.eval()

    # Vocabulary
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    return caption_model, seg_model, vocab

def predict_caption(img_tensor, model, vocab, max_len=20):
    img = img_tensor.unsqueeze(0).to(device)
    tgt = torch.tensor([[vocab.stoi["<start>"]]], dtype=torch.long).to(device)
    caption = []

    with torch.no_grad():
        for _ in range(max_len):
            output = model(img, tgt)
            next_word = output.argmax(2)[:, -1].item()
            if next_word == vocab.stoi["<end>"]:
                break
            caption.append(vocab.itos.get(next_word, "<unk>"))
            tgt = torch.cat([tgt, torch.tensor([[next_word]], dtype=torch.long).to(device)], dim=1)

    return " ".join(caption)

def predict_segmentation(img_tensor, model):
    with torch.no_grad():
        img = img_tensor.unsqueeze(0).to(device)
        output = model(img)[0]
    return output['masks']
