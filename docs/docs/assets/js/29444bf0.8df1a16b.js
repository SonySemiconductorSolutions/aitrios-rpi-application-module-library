"use strict";(self.webpackChunkmodlib_docs=self.webpackChunkmodlib_docs||[]).push([[968],{1911:(e,s,r)=>{r.r(s),r.d(s,{assets:()=>a,contentTitle:()=>d,default:()=>h,frontMatter:()=>l,metadata:()=>n,toc:()=>o});const n=JSON.parse('{"id":"devices/interpreters/index","title":"Interpreters","description":"KerasInterpreter","source":"@site/docs-api/devices/interpreters/index.md","sourceDirName":"devices/interpreters","slug":"/devices/interpreters/","permalink":"/docs/api-reference/devices/interpreters/","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedAt":1732028012000,"sidebarPosition":3,"frontMatter":{"title":"Interpreters","sidebar_position":3},"sidebar":"tutorialSidebar","previous":{"title":"Frame","permalink":"/docs/api-reference/devices/frame"},"next":{"title":"Sources","permalink":"/docs/api-reference/devices/sources"}}');var t=r(4848),i=r(8453);const l={title:"Interpreters",sidebar_position:3},d=void 0,a={},o=[{value:"KerasInterpreter",id:"kerasinterpreter",level:2},{value:"Methods",id:"methods",level:3},{value:'<span class="signature-title">__init__</span>',id:"__init__",level:4},{value:'<span class="signature-title">deploy</span>',id:"deploy",level:4},{value:'<span class="signature-title">__enter__</span>',id:"__enter__",level:4},{value:'<span class="signature-title">__exit__</span>',id:"__exit__",level:4},{value:'<span class="signature-title">__iter__</span>',id:"__iter__",level:4},{value:'<span class="signature-title">__next__</span>',id:"__next__",level:4},{value:'<span class="signature-title">load_tf_keras_model</span>',id:"load_tf_keras_model",level:4}];function c(e){const s={a:"a",br:"br",code:"code",h2:"h2",h3:"h3",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,i.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("div",{className:"module-separator","data-content":"Class"}),"\n",(0,t.jsx)(s.h2,{id:"kerasinterpreter",children:"KerasInterpreter"}),"\n",(0,t.jsxs)(s.p,{children:[(0,t.jsxs)("div",{className:"signature",children:[(0,t.jsx)("b",{children:"KerasInterpreter("}),"source:\xa0",(0,t.jsx)(s.a,{href:"/api-reference/devices/sources#source",children:"Source"}),", headless:\xa0Optional[bool]\xa0=\xa0False, timeout:\xa0Optional[int]\xa0=\xa0None",(0,t.jsx)("b",{children:")"})]}),"\nKeras Interpreter device."]}),"\n",(0,t.jsx)(s.p,{children:"This device module allows to run inference of Keras models locally and is designed for test/development purposes.\nOutput tensors are post-processed by the model post-processor function and attached to the frame."}),"\n",(0,t.jsx)(s.p,{children:"Example:"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{children:"from modlib.devices import KerasInterpreter\n\ndevice = KerasInterpreter()\nmodel = CustomKerasModel(...)\ndevice.deploy(model)\n\nwith device as stream:\n    for frame in stream:\n        print(frame.detections)\n"})}),"\n",(0,t.jsx)(s.p,{children:(0,t.jsx)(s.strong,{children:"Inherites"})}),"\n",(0,t.jsxs)(s.ul,{children:["\n",(0,t.jsx)(s.li,{children:(0,t.jsx)(s.a,{href:"/api-reference/devices/device#device",children:"Device"})}),"\n"]}),"\n",(0,t.jsx)(s.h3,{id:"methods",children:"Methods"}),"\n",(0,t.jsx)(s.h4,{id:"__init__",children:(0,t.jsx)("span",{className:"signature-title",children:"__init__"})}),"\n",(0,t.jsxs)("div",{className:"signature",children:[(0,t.jsx)("b",{children:"__init__("}),"self, source:\xa0",(0,t.jsx)(s.a,{href:"/api-reference/devices/sources#source",children:"Source"}),", headless:\xa0Optional[bool]\xa0=\xa0False, timeout:\xa0Optional[int]\xa0=\xa0None",(0,t.jsx)("b",{children:")"})]}),"\n",(0,t.jsx)(s.p,{children:"Initialize a Keras Interpreter device."}),"\n",(0,t.jsx)(s.p,{children:(0,t.jsx)(s.strong,{children:"Args:"})}),"\n",(0,t.jsxs)(s.ul,{children:["\n",(0,t.jsxs)(s.li,{children:[(0,t.jsx)(s.strong,{children:"source"})," (",(0,t.jsx)(s.a,{href:"/api-reference/devices/sources#source",children:"Source"}),"): The source of the Keras model."]}),"\n",(0,t.jsxs)(s.li,{children:[(0,t.jsx)(s.strong,{children:"headless"})," (Optional[bool]\xa0=\xa0False): Whether to run the interpreter in headless mode. Defaults to False."]}),"\n",(0,t.jsxs)(s.li,{children:[(0,t.jsx)(s.strong,{children:"timeout"})," (Optional[int]\xa0=\xa0None): The timeout value for the interpreter. Defaults to None."]}),"\n"]}),"\n",(0,t.jsx)(s.h4,{id:"deploy",children:(0,t.jsx)("span",{className:"signature-title",children:"deploy"})}),"\n",(0,t.jsxs)("div",{className:"signature",children:[(0,t.jsx)("b",{children:"deploy("}),"self, model:\xa0",(0,t.jsx)(s.a,{href:"/api-reference/models/model#model",children:"Model"}),(0,t.jsx)("b",{children:")"})]}),"\n",(0,t.jsx)(s.p,{children:"Deploys a Keras model for local inference."}),"\n",(0,t.jsx)(s.p,{children:(0,t.jsx)(s.strong,{children:"Args:"})}),"\n",(0,t.jsxs)(s.ul,{children:["\n",(0,t.jsxs)(s.li,{children:[(0,t.jsx)(s.strong,{children:"model"})," (",(0,t.jsx)(s.a,{href:"/api-reference/models/model#model",children:"Model"}),"): The Keras model to deploy."]}),"\n"]}),"\n",(0,t.jsxs)(s.p,{children:[(0,t.jsx)(s.strong,{children:"Raises:"}),(0,t.jsx)(s.br,{}),"\n","FileNotFoundError: If the model file is not found.",(0,t.jsx)(s.br,{}),"\n","TypeError: If the model or model_file is not a Keras model.",(0,t.jsx)(s.br,{}),"\n","AttributeError: If the model does not have a pre_process method."]}),"\n",(0,t.jsx)(s.h4,{id:"__enter__",children:(0,t.jsx)("span",{className:"signature-title",children:"__enter__"})}),"\n",(0,t.jsxs)("div",{className:"signature",children:[(0,t.jsx)("b",{children:"__enter__("}),"self",(0,t.jsx)("b",{children:")"})]}),"\n",(0,t.jsx)(s.p,{children:"Start the KerasInterpreter device stream."}),"\n",(0,t.jsx)(s.h4,{id:"__exit__",children:(0,t.jsx)("span",{className:"signature-title",children:"__exit__"})}),"\n",(0,t.jsxs)("div",{className:"signature",children:[(0,t.jsx)("b",{children:"__exit__("}),"self, exc_type, exc_val, exc_tb",(0,t.jsx)("b",{children:")"})]}),"\n",(0,t.jsx)(s.p,{children:"Stop the KerasInterpreter device stream."}),"\n",(0,t.jsx)(s.h4,{id:"__iter__",children:(0,t.jsx)("span",{className:"signature-title",children:"__iter__"})}),"\n",(0,t.jsxs)("div",{className:"signature",children:[(0,t.jsx)("b",{children:"__iter__("}),"self",(0,t.jsx)("b",{children:")"})]}),"\n",(0,t.jsx)(s.p,{children:"Iterate over the frames in the device stream."}),"\n",(0,t.jsx)(s.h4,{id:"__next__",children:(0,t.jsx)("span",{className:"signature-title",children:"__next__"})}),"\n",(0,t.jsxs)("div",{className:"signature",children:[(0,t.jsx)("b",{children:"__next__("}),"self",(0,t.jsx)("b",{children:")"})]}),"\n",(0,t.jsx)(s.p,{children:"Get the next frame in the device stream."}),"\n",(0,t.jsxs)(s.p,{children:[(0,t.jsx)(s.strong,{children:"Returns:"}),(0,t.jsx)(s.br,{}),"\n","The next frame in the device stream."]}),"\n",(0,t.jsx)(s.h4,{id:"load_tf_keras_model",children:(0,t.jsx)("span",{className:"signature-title",children:"load_tf_keras_model"})}),"\n",(0,t.jsx)("div",{className:"decorator",children:"@staticmethod"}),"\n",(0,t.jsxs)("div",{className:"signature",children:[(0,t.jsx)("b",{children:"load_tf_keras_model("}),"model_path:\xa0str",(0,t.jsx)("b",{children:")"})]}),"\n",(0,t.jsxs)(s.p,{children:["Loads the keras model file as a ",(0,t.jsx)(s.code,{children:"tf.keras.model"}),".",(0,t.jsx)(s.br,{}),"\n","Requires tensorflow 2.14 to be installed."]}),"\n",(0,t.jsxs)(s.p,{children:[(0,t.jsx)(s.strong,{children:"Raises:"}),(0,t.jsx)(s.br,{}),"\n","ImportError: When loading the model fails due to missing tensorflow dependency."]})]})}function h(e={}){const{wrapper:s}={...(0,i.R)(),...e.components};return s?(0,t.jsx)(s,{...e,children:(0,t.jsx)(c,{...e})}):c(e)}},8453:(e,s,r)=>{r.d(s,{R:()=>l,x:()=>d});var n=r(6540);const t={},i=n.createContext(t);function l(e){const s=n.useContext(i);return n.useMemo((function(){return"function"==typeof e?e(s):{...s,...e}}),[s,e])}function d(e){let s;return s=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:l(e.components),n.createElement(i.Provider,{value:s},e.children)}}}]);