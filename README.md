# Type Safe OpenCV

A simple wrapper around opencv to make it type-safe and also follow pythonic syntax
using builder pattern and similar systems.

## Example

<table>
<tr>
<th>The old way: raw <code>cv2</code></th>
<th>The builder pattern: <code>cv4</code></th>
</tr>
<tr>
<td>

```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# nothing stops you passing the HSV image here
# by mistake it runs, but the result is very wrong
contours, _ = cv2.findContours(
    hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    if cv2.contourArea(c) < MIN_AREA:
        continue
    m = cv2.moments(c)
    cx = m["m10"] / m["m00"]  # can divide by zero
```

</td>
<td>

```python
contours = (
    hsv.mask(BLUE)
       .clean(5)
       .find_contours()
       .filter(lambda c: c.area > MIN_AREA)
       .filter(lambda c: c.base_y >= MIN_Y)
)

# find_contours() lives ONLY on HSVMask.
# hsv.find_contours() does not exist: the
# editor won't autocomplete it, and the type
# checker rejects it before you ever run.

for c in contours:
    cx = c.cx        # centroid, typed
    base = c.base_y  # ground contact
```

</td>
</tr>
</table>

Where `hsv` and `BLUE` come from:

```python
from cv4 import BGRImage, HSV, HSVRange

BLUE = HSVRange(lower=HSV(100, 150, 50), upper=HSV(130, 255, 255))
hsv = BGRImage(frame).as_hsv()  # frame: a BGR image from cv2
```
