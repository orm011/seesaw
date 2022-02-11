export function image_accepted(imdata){
  return imdata.boxes !== null ? imdata.boxes.filter(b => b.marked_accepted).length > 0 : false;
}