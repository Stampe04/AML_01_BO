import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG16(torch.nn.Module):
    def __init__(self, num_classes, num_kernels=64, dropout_rate=0.5, in_channels=1, dataset=None):
        super().__init__()
        
        # Helper hyperparameters to keep track of VGG16 architecture
        conv_stride = 1
        pool_stride = 2
        conv_kernel = 3
        pool_kernel = 2
        dropout_probs = dropout_rate
        
        # Define features and classifier each individually, this is how the VGG16-D model is orignally defined
        self.features = torch.nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_kernels, kernel_size=conv_kernel, padding=1, stride=conv_stride), # dim = in
            nn.ReLU(),
            nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels, kernel_size=conv_kernel, padding=1, stride=conv_stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride= pool_stride),

            nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels*2, kernel_size=conv_kernel, padding=1, stride=conv_stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_kernels*2, out_channels=num_kernels*2, kernel_size=conv_kernel, padding=1, stride=conv_stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride= pool_stride),


            nn.Flatten(),
        ).to(device)
        
        self.classifier = torch.nn.Sequential(
            nn.LazyLinear(out_features=600),
            nn.ReLU(),
            nn.Dropout(dropout_probs),
            nn.Linear(in_features=600, out_features=120),
            nn.ReLU(),
            nn.Dropout(dropout_probs),
            nn.Linear(in_features=120, out_features=num_classes),

        ).to(device)
        
        # In the paper, they mention updating towards the 'multinomial logistic regression objective'
        # As can be read in Bishop p. 159, taking the logarithm of this equates to the cross-entropy loss function.
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer - For now just set to Adam to test the implementation
        self.optim = torch.optim.Adam(
            list(self.features.parameters()) + list(self.classifier.parameters()),
            lr=1e-3,
            weight_decay=1e-4,
        )
        # self.optim = torch.optim.SGD(list(self.features.parameters()) + list(self.classifier.parameters()), lr=learning_rate, momentum=optim_momentum, weight_decay=weight_decay)

        # learning rate scheduler:
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=10, gamma=0.1)

        self.dataset = dataset

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
