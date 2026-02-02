# Manim Slides for Convex Optimization

## What is Manim Slides?

Manim Slides is a tool that turns Manim animations into interactive presentations. Unlike embedding videos in PDFs (which is unreliable), Manim Slides creates:

1. **HTML presentations** (Reveal.js) - Works in any browser, including iPad Safari!
2. **Native GUI presentations** - For desktop use
3. **PowerPoint export** - Via the HTML → PPTX conversion

## Files Included

- `convex_optimization_presentation.html` - The main presentation file
- `convex_optimization_presentation_assets/` - Video files for each slide
- `convex_slides.py` - Source code to regenerate/modify

## How to Use on iPad

1. **Unzip** the package
2. **Transfer** both the `.html` file AND the `_assets` folder to your iPad (keep them in the same directory!)
3. **Open** the HTML file in Safari
4. **Navigate** with swipes or tap edges of screen

## Controls

| Action | Desktop | iPad |
|--------|---------|------|
| Next slide | Space / → / Click | Swipe left / Tap right |
| Previous slide | ← | Swipe right / Tap left |
| Overview | O | Two-finger pinch |
| Fullscreen | F | N/A |

## How to Modify

### Prerequisites
```bash
pip install manim manim-slides
```

### Edit the source
Modify `convex_slides.py` - key concepts:

```python
from manim import *
from manim_slides import Slide

class MySlide(Slide):
    def construct(self):
        # Your animations here
        self.play(Write(text))
        
        self.next_slide()  # Creates a pause point!
        
        # More animations
        self.play(Create(circle))
        
        self.next_slide()
```

### Render
```bash
manim-slides render convex_slides.py MySlide
```

### Present (desktop)
```bash
manim-slides present MySlide
```

### Export to HTML (for iPad)
```bash
manim-slides convert MySlide presentation.html -ccontrols=true
```

## Tips for iPad Presentations

1. **Use Safari** - Other browsers may have video playback issues
2. **Keep files together** - The HTML and assets folder must be in the same directory
3. **Test offline** - Download assets beforehand for reliable playback
4. **Landscape mode** - Presentations look best in landscape

## Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Manim Slides HTML** | Works on iPad, interactive | Need to transfer assets folder |
| **Beamer + static images** | Works in Notability | No animation |
| **Beamer + keyframes** | Works in Notability, step-by-step | Manual, no smooth animation |
| **Beamer + animate package** | Smooth animation | Only works in Adobe Reader |
| **PowerPoint/Keynote** | Native video support | Less math typesetting control |

## Resources

- [Manim Slides Documentation](https://manim-slides.eertmans.be/)
- [Manim Community](https://www.manim.community/)
- [Reveal.js (underlying HTML framework)](https://revealjs.com/)
