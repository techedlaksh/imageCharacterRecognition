for file in *.jpg; do convert -resize 720x720 -- "$file" "./${file%%.jpg}-resized.jpg"; done

