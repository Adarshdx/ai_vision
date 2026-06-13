import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from pathlib import Path
from config import Config
from model import CIFAR10Classifier

def load_model_for_inference(model_path):
    """Load trained model for inference"""
    # Create model
    model = CIFAR10Classifier(num_classes=Config.NUM_CLASSES, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    print(f"Best validation accuracy: {max(checkpoint['history']['val_acc']):.2f}%")
    
    return model

def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(Config.MEAN, Config.STD)
    ])
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def predict_image(model, image_tensor, return_probabilities=False):
    """Predict class for a single image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(Config.DEVICE)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        predicted_class = Config.CLASSES[predicted.item()]
        confidence = probabilities[0][predicted.item()].item() * 100
    
    if return_probabilities:
        all_probs = {Config.CLASSES[i]: probabilities[0][i].item() * 100 
                    for i in range(len(Config.CLASSES))}
        return predicted_class, confidence, all_probs
    
    return predicted_class, confidence

def predict_batch(model, image_paths):
    """Predict for multiple images"""
    results = []
    for image_path in image_paths:
        image_tensor = preprocess_image(image_path)
        predicted_class, confidence = predict_image(model, image_tensor)
        results.append({
            'image': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence
        })
    return results

def main():
    parser = argparse.ArgumentParser(description='Run inference on CIFAR-10 classifier')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, nargs='+', help='Paths to multiple images')
    parser.add_argument('--show_probs', action='store_true', 
                       help='Show all class probabilities')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first using: python train.py")
        return
    
    # Load model
    model = load_model_for_inference(model_path)
    
    # Single image inference
    if args.image:
        print(f"\nPredicting for image: {args.image}")
        image_tensor = preprocess_image(args.image)
        
        if args.show_probs:
            pred_class, confidence, all_probs = predict_image(
                model, image_tensor, return_probabilities=True
            )
            print(f"\nPredicted Class: {pred_class}")
            print(f"Confidence: {confidence:.2f}%\n")
            print("All Class Probabilities:")
            for class_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name:12s}: {prob:6.2f}%")
        else:
            pred_class, confidence = predict_image(model, image_tensor)
            print(f"\nPredicted Class: {pred_class}")
            print(f"Confidence: {confidence:.2f}%")
    
    # Batch inference
    elif args.batch:
        print(f"\nPredicting for {len(args.batch)} images...")
        results = predict_batch(model, args.batch)
        print("\nResults:")
        print("-" * 60)
        for result in results:
            print(f"{Path(result['image']).name:30s} -> {result['predicted_class']:12s} "
                  f"(confidence: {result['confidence']:.2f}%)")
    
    else:
        print("Please provide either --image or --batch argument")
        print("\nExample usage:")
        print("  python inference.py --image path/to/image.jpg")
        print("  python inference.py --batch img1.jpg img2.jpg img3.jpg")
        print("  python inference.py --image image.jpg --show_probs")

if __name__ == "__main__":
    main()