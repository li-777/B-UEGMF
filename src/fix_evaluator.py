import re

def fix_evaluator_file(file_path='evaluator.py'):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()  
    
    fixes = [
        (r'\.cpu\(\)\.numpy\(\)', '.cpu().detach().numpy()'),
        
        (r'\.permute\([^)]+\)\.cpu\(\)\.numpy\(\)', 
         lambda m: m.group(0).replace('.cpu().numpy()', '.cpu().detach().numpy()')),
        
        (r'\.flatten\(\)\.cpu\(\)\.numpy\(\)', '.flatten().cpu().detach().numpy()'),
    ]

    for pattern, replacement in fixes:
        if callable(replacement):
            content = re.sub(pattern, replacement, content)
        else:
            content = re.sub(pattern, replacement, content)
    
    plot_image_fix = """
    def _plot_image_comparison(self, clean_images: torch.Tensor,
                              unlearnable_images: torch.Tensor,
                              perturbations: torch.Tensor,
                              experiment_name: str) -> None:
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        
        for i in range(5):
            clean_img = clean_images[i].permute(1, 2, 0).cpu().detach().numpy()
            clean_img = np.clip(clean_img, 0, 1)
            axes[0, i].imshow(clean_img)
            axes[0, i].set_title(f'Clean {i+1}')
            axes[0, i].axis('off')

            unlearnable_img = unlearnable_images[i].permute(1, 2, 0).cpu().detach().numpy()
            unlearnable_img = np.clip(unlearnable_img, 0, 1)
            axes[1, i].imshow(unlearnable_img)
            axes[1, i].set_title(f'Unlearnable {i+1}')
            axes[1, i].axis('off')
            
            perturbation = perturbations[i].permute(1, 2, 0).cpu().detach().numpy()
            perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
            axes[2, i].imshow(perturbation)
            axes[2, i].set_title(f'Perturbation {i+1}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{experiment_name}_image_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    """
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"{file_path} ")

MANUAL_FIX_GUIDE = """
python -c "
import re
with open('evaluator.py', 'r') as f: content = f.read()
content = re.sub(r'\.cpu\(\)\.numpy\(\)', '.cpu().detach().numpy()', content)
with open('evaluator.py', 'w') as f: f.write(content)
"
"""

if __name__ == "__main__":
    print(MANUAL_FIX_GUIDE)