export function image_accepted(imdata){
  return imdata.boxes.filter(b => b.marked_accepted).length > 0
}