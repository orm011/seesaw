import {Imdata} from './basic_types'

export function image_accepted(imdata : Imdata) : boolean {
  return imdata.boxes ? imdata.boxes.filter(b => b.marked_accepted).length > 0 : false;
}