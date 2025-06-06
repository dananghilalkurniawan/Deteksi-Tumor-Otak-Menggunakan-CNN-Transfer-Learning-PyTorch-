# Visualisasi Hasil Akhir
def imshow(inp, title=None):
    """Imshow untuk tensor gambar, normalisasi dibalik."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')

def visualize_model(model, dataloader, class_names, device, num_images=6):
    model.eval()
    images_so_far = 0

    plt.figure(figsize=(20, 15))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                plt.subplot(num_images//3, 3, images_so_far)
                title = f'Pred: {class_names[preds[j]]}\nActual: {class_names[labels[j]]}'
                imshow(inputs.cpu().data[j], title=title)

                if images_so_far == num_images:
                    plt.show()
                    return
    plt.show()
visualize_model(model_trained, dataloaders['val'], class_names, device, num_images=12)
