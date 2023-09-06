class GalaxySimCLR(nn.Module):
    def __init__(self,premodel,num_classes):
        super().__init__()

        self.premodel = premodel
        self.num_classes = num_classes

        for p in self.premodel.parameters():
            p.requires_grad = False

        for p in self.premodel.projector.parameters():
            p.requires_grad = False

        self.lastlayer = nn.Linear(2048,self.num_classes)

    def forward(self,x):
        out = self.premodel.pretrained(x)
        out = self.lastlayer(out)
        return out
