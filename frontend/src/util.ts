import {Imdata} from './basic_types'

export function image_accepted(imdata : Imdata) : boolean {
  return imdata.boxes ? imdata.boxes.filter(b => b.marked_accepted).length > 0 : false;
}


// https://www.w3schools.com/js/js_cookies.asp
export function setCookie(cname, cvalue, exdays) {
  const d = new Date();
  d.setTime(d.getTime() + (exdays*24*60*60*1000));
  let expires = "expires="+ d.toUTCString();
  document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/";
}

export function getCookie(cname) {
  let name = cname + "=";
  let decodedCookie = decodeURIComponent(document.cookie);
  let ca = decodedCookie.split(';');
  for(let i = 0; i <ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) == ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) == 0) {
      return c.substring(name.length, c.length);
    }
  }
  return null;
}

// // https://stackoverflow.com/questions/10730362/get-cookie-by-name/15724300#15724300
// export function getCookie(name) {
//   const value = `; ${document.cookie}`;
//   const parts = value.split(`; ${name}=`);
//   if (parts.length === 2) { 
//     return parts.pop().split(';').shift();
//   } 
//
