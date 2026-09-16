import AppKit
let w = 1400, h = 1000
let image = NSImage(size: NSSize(width: w, height: h)); image.lockFocus()
NSColor.white.setFill(); NSRect(x:0,y:0,width:w,height:h).fill()
func text(_ s:String,_ x:CGFloat,_ y:CGFloat,_ size:CGFloat,_ color:NSColor = .black,_ bold:Bool=false) { let f = NSFont(name: bold ? "Helvetica-Bold" : "Helvetica", size:size)!; s.draw(at:NSPoint(x:x,y:y), withAttributes:[.font:f,.foregroundColor:color]) }
text("NVDA News Themes Around Price Events",70,930,32,.black,true); text("Mean share of matched articles; labels show Up / Down percentages",70,895,18,.gray)
let labels=["AI infrastructure","Stock/investor commentary","AMD/competitors & deals","China/export controls","Market-wide coverage","Jensen Huang/CEO","Micron/HBM memory","NVDA commentary/media"]
let up=[25.8,24.9,12.0,10.5,5.3,6.4,7.2,9.3], down=[25.1,24.1,11.2,11.9,5.1,7.6,6.1,9.0]
for i in 0..<labels.count { let y=820-CGFloat(i*55); text(labels[i],70,y,17,.darkGray); NSColor(calibratedRed:0.90,green:0.91,blue:0.93,alpha:1).setFill(); NSRect(x:330,y:y-2,width:900,height:18).fill(); NSColor.systemBlue.setFill(); NSRect(x:330,y:y-2,width:CGFloat(up[i])*30,height:18).fill(); NSColor.systemRed.setFill(); NSRect(x:330,y:y-24,width:CGFloat(down[i])*30,height:8).fill(); text(String(format:"%.1f / %.1f%%",up[i],down[i]),1245,y,15,.gray) }
text("What each topic contains",70,355,24,.black,true)
let words=["AI, infrastructure, data center, cloud, Microsoft, Amazon","stock, buy, investors, Wall Street, shares, earnings","AMD, OpenAI, Meta, deal, Broadcom, AI chip","China, H200, chips, Trump, Chinese, export","Dow, stock market, market today, futures, index","Jensen Huang, CEO, Nvidia CEO, OpenAI","Micron, memory, HBM, bandwidth, Samsung, supply","Jim Cramer, Nasdaq, stocks, NVDA"]
for i in 0..<labels.count { let y=315-CGFloat(i*34); text(labels[i],70,y,15,.darkGray,true); text(words[i],330,y,15,.darkGray) }
image.unlockFocus(); let data=image.tiffRepresentation!; let rep=NSBitmapImageRep(data:data)!; try! rep.representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:"Theme_analysis/output/theme_analysis_summary.png"))
