"use strict";(self.webpackChunkmodlib_docs=self.webpackChunkmodlib_docs||[]).push([[941],{8251:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>s,contentTitle:()=>a,default:()=>p,frontMatter:()=>l,metadata:()=>i,toc:()=>d});const i=JSON.parse('{"id":"getting_started/hello_world","title":"Hello world","description":"Setup your device","source":"@site/docs/getting_started/hello_world.md","sourceDirName":"getting_started","slug":"/getting_started/hello_world","permalink":"/docs/getting_started/hello_world","draft":false,"unlisted":false,"tags":[],"version":"current","lastUpdatedAt":1732530754000,"sidebarPosition":0,"frontMatter":{"title":"Hello world","sidebar_position":0},"sidebar":"tutorialSidebar","previous":{"title":"Getting Started","permalink":"/docs/category/getting-started"},"next":{"title":"Model zoo","permalink":"/docs/getting_started/model_zoo"}}');var r=n(4848),o=n(8453);const l={title:"Hello world",sidebar_position:0},a="Getting Started",s={},d=[{value:"Setup your device",id:"setup-your-device",level:2},{value:"Install the Application Module Library",id:"install-the-application-module-library",level:2},{value:"Verify your setup",id:"verify-your-setup",level:2},{value:"First example",id:"first-example",level:2}];function c(e){const t={a:"a",admonition:"admonition",br:"br",code:"code",h1:"h1",h2:"h2",header:"header",li:"li",mdxAdmonitionTitle:"mdxAdmonitionTitle",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,o.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(t.header,{children:(0,r.jsx)(t.h1,{id:"getting-started",children:"Getting Started"})}),"\n",(0,r.jsx)(t.h2,{id:"setup-your-device",children:"Setup your device"}),"\n",(0,r.jsxs)(t.admonition,{type:"info",children:[(0,r.jsx)(t.mdxAdmonitionTitle,{}),(0,r.jsxs)(t.p,{children:["The Application Module Library currently supports the ",(0,r.jsx)(t.strong,{children:"Raspberry Pi AI Camera"})," only."]})]}),"\n",(0,r.jsx)(t.p,{children:"The Raspberry Pi AI Camera is an extremely capable piece of hardware, enabling you to build powerful AI applications on your Raspberry Pi. By offloading the AI inference to the IMX500 accelerator chip, more computational resources are available to handle application logic right on the edge!"}),"\n",(0,r.jsxs)(t.p,{children:["If you haven't done so already, make sure to ",(0,r.jsx)(t.a,{href:"https://www.raspberrypi.com/documentation/accessories/ai-camera.html",children:"verify"})," that your AI Camera is set up correctly."]}),"\n",(0,r.jsx)(t.h2,{id:"install-the-application-module-library",children:"Install the Application Module Library"}),"\n",(0,r.jsx)(t.p,{children:"Install the Application Module Library in your current Python (virtual) environment."}),"\n",(0,r.jsxs)(t.ol,{children:["\n",(0,r.jsx)(t.li,{children:"Ensure that your Raspberry Pi runs the latest software:"}),"\n"]}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-shell",children:"sudo apt update && sudo apt full-upgrade\nsudo apt install imx500-all\nsudo apt install python3-opencv python3-munkres python3-picamera2\n"})}),"\n",(0,r.jsxs)(t.ol,{start:"2",children:["\n",(0,r.jsx)(t.li,{children:"(Optional) Create and enable a virtual environment."}),"\n"]}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-shell",children:"python -m venv .venv --system-site-packages\nsource .venv/bin/activate\n"})}),"\n",(0,r.jsxs)(t.ol,{start:"3",children:["\n",(0,r.jsx)(t.li,{children:"One can use pip to install the library in your project Python environment."}),"\n"]}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-shell",children:"pip install git+https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git\n"})}),"\n",(0,r.jsx)(t.h2,{id:"verify-your-setup",children:"Verify your setup"}),"\n",(0,r.jsxs)(t.p,{children:["Let's verify that our camera is connected and the Application Module Library is installed correctly.",(0,r.jsx)(t.br,{}),"\n","Create a new Python file named ",(0,r.jsx)(t.code,{children:"hello_world.py"}),". and run the following code to see a camera preview."]}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-python",metastring:'title="hello_world.py"',children:"from modlib.devices import AiCamera\n\ndevice = AiCamera()\n\nwith device as stream:\n    for frame in stream:\n        frame.display()\n"})}),"\n",(0,r.jsx)(t.h2,{id:"first-example",children:"First example"}),"\n",(0,r.jsx)(t.p,{children:"Let's dive into our first example! This example will demonstrate how to use the AI Camera to detect objects in real-time using a pre-trained SSDMobileNet model."}),"\n",(0,r.jsxs)(t.p,{children:["First, extend the ",(0,r.jsx)(t.code,{children:"hello_world.py"})," example in your project directory and add the following code:"]}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-python",metastring:'title="hello_world.py"',children:'from modlib.apps import Annotator\nfrom modlib.devices import AiCamera\nfrom modlib.models.zoo import SSDMobileNetV2FPNLite320x320\n\ndevice = AiCamera()\nmodel = SSDMobileNetV2FPNLite320x320()\ndevice.deploy(model)\n\nannotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)\n\nwith device as stream:\n    for frame in stream:\n        detections = frame.detections[frame.detections.confidence > 0.55]\n        labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]\n        \n        annotator.annotate_boxes(frame, detections, labels=labels)\n        frame.display()\n'})}),"\n",(0,r.jsx)(t.p,{children:"A brief overview of the key steps in this example:"}),"\n",(0,r.jsxs)(t.ul,{children:["\n",(0,r.jsxs)(t.li,{children:["Initiate the ",(0,r.jsx)(t.code,{children:"AiCamera"}),"device"]}),"\n",(0,r.jsxs)(t.li,{children:["Initiate the pre-packaged ",(0,r.jsx)(t.code,{children:"SSDMobileNetV2FPNLite320x320"})," model from the Zoo."]}),"\n",(0,r.jsx)(t.li,{children:"Deploy the model to the device."}),"\n",(0,r.jsx)(t.li,{children:"Start the stream and visualize the detections that have a confidence greater then the given threshold (0.55)."}),"\n"]})]})}function p(e={}){const{wrapper:t}={...(0,o.R)(),...e.components};return t?(0,r.jsx)(t,{...e,children:(0,r.jsx)(c,{...e})}):c(e)}},8453:(e,t,n)=>{n.d(t,{R:()=>l,x:()=>a});var i=n(6540);const r={},o=i.createContext(r);function l(e){const t=i.useContext(o);return i.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function a(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:l(e.components),i.createElement(o.Provider,{value:t},e.children)}}}]);