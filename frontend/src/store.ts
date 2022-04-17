import { reactive } from 'vue'
import { Imdata, Box } from './basic_types'


export const store = reactive({ 
    _label_db : new Map() as Map<Number,Imdata>,

    addImdata(imdata : Imdata){
        this._label_db.set(imdata.dbidx, imdata);
    },

    getImdata(dbidx : Number) {
        return this._label_db.get(dbidx);
    },
});