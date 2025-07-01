import torch

class DiceLoss(torch.nn.Module):
    def __init__(self, convert_logits_to_probs=False, reduce=True):
        super().__init__()
        self.convert_logits_to_probs = convert_logits_to_probs
        self.reduce = reduce
    
    def forward(self, ground_truth, input):
        """dice loss for single class segmentation"""

        # sigmoid
        if self.convert_logits_to_probs:
            input = torch.sigmoid(input)

        if input.dim() == 5:
            input.squeeze(0)
        
        
        
        intersection = (input*ground_truth).sum(dim=(1,2,3))
        union = (input + ground_truth).sum(dim=(1,2,3))
        dice = 0 if union.sum() == 0 else 2*intersection/union
        
        if self.reduce:
            dice = torch.mean(dice)
            
        loss = (1 - dice)
        

        return loss