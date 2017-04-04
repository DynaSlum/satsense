% dragable & resizeable & unit-support %menucommand-support  SCALEBAR
% @Chenxinfeng, 2016-9-10
%
% ================================HOW TO USE==============================
% ----PREPARE---
% plot(sin(1:0.1:10));
% obj = scalebar; %default, recommanded
% obj = scalebar(___,Name,Value);
% obj = scalebar(ax,___);
%
% ---GUI support---
% %drag the SCALE-LABEL, to move only the LABEL position.
% %drag the SCALE-LINE, to move the whole SCALE position.
% %Right-Click on SCALE-LABEL-Y, to ROTATE this LABEL.
% %Right-Click on SCALE-LINE, to set SCALE-Length and -Unit.
%
% ---Command support---
% obj.XLen = 15;              %X-Length, 15.
% obj.XUnit = 'm';            %X-Unit, 'm'.
% obj.Position = [55, -0.6];  %move the whole SCALE position.
% obj.hTextX_Pos = [1,-0.06]; %move only the LABEL position
% obj.Border = 'UL';          %'LL'(default), 'LR', 'UL', 'UR'
% obj.hTextY_Rot = 0;         %Y-LABEL rotation change as horizontal.

classdef scalebar <handle
    properties (SetAccess = private, GetAccess = public)
	% GUI objects
		hLineX %SCALE-X-LINE, L&R
		hLineY %SCALE-Y-LINE, L&H
		hTextX %SCALE-X-LABEL
		hTextY %SCALE-Y-LABEL
	end
	properties (SetObservable=true)
	% Main properties
		Position              %SCALE-POSITION, [X,Y]
		Border='LL'           %'LL', 'LR', 'UL', 'UR'
		XUnit=''              %SCALE-X-UNIT, string
		YUnit=''              %SCALE-Y-UNIT
		XLen                  %SCALE-X-LENGTH
		YLen                  %SCALE-Y-LENGTH
		hTextX_Pos            %SCALE-X-LABEL-POSITION
		hTextY_Pos            %SCALE-Y-LABEL-POSITION
		hTextY_Rot=90         %SCALE-Y-LABEL-ROTATION
	end
	methods
		function hobj = scalebar(varargin)                        
            %get axes
            if isempty(varargin)||~isscalar(varargin{1})||~ishandle(varargin{1})
                hAxes = gca;
            else
                hAxes = varargin{1};
            end
            hold(hAxes,'on'); 
            %props pretend initing
            hobj.XLen = 0;
            hobj.YLen = 0;
            hobj.Position = [0 0];
            hobj.hTextX_Pos = [0 0];
            hobj.hTextY_Pos = [0 0];
            
            %listen to Prop change
            for prop={'XLen','YLen','XUnit','YUnit','hTextY_Rot'...
                      'Position','Border','hTextX_Pos','hTextY_Pos'}
                funstr = eval(['@hobj.Set',prop{1}]);
                addlistener(hobj,prop{1},'PostSet',funstr);
            end
			%Get axis parameters
			axisXLim = get(hAxes,'XLim');
			axisYLim = get(hAxes,'YLim');
			axisXWidth = diff(axisXLim);
			axisYWidth = diff(axisYLim);

			%Get or Create GUI handles
			templine = plot([0 0],[0 0],'Parent',hAxes,'Color','k','LineWidth',1.5);
            hobj.hLineX = [copy(templine), templine];
			hobj.hLineY = [copy(templine), copy(templine)];
            set([hobj.hLineY, hobj.hLineX], 'Parent',hAxes,'ButtonDownFcn',@hobj.FcnStartDrag); 
			hobj.hTextX = text(0,0,'','Parent',hAxes,'ButtonDownFcn',@hobj.FcnStartDrag);
			hobj.hTextY = text(0,0,'','Parent',hAxes,'Rotation',90,'ButtonDownFcn',@hobj.FcnStartDrag);
            
            %UIMENU for RITHT-CLICK
            hcmenu = uicontextmenu;
            set([hobj.hLineX, hobj.hLineY],'uicontextmenu',hcmenu);
            uimenu('parent',hcmenu,'label','[X] Length', ...
                   'callback',@(o,e)hobj.uiSetLen('X'));
            uimenu('parent',hcmenu,'label','[X] Unit', ...
                   'callback',@(o,e)hobj.uiSetUnit('X'));
            uimenu('parent',hcmenu,'label','[Y] Length', 'separator','on',...
                   'callback',@(o,e)hobj.uiSetLen('Y'));
            uimenu('parent',hcmenu,'label','[Y] Unit', ...
                   'callback',@(o,e)hobj.uiSetUnit('Y'));
            uimenu('parent',hcmenu,'label','Delete me', 'separator','on',...
                   'callback',@(o,e)hobj.delete(),'ForegroundColor',[0,0,1]);
            hcmenu_text = uicontextmenu;
            set(hobj.hTextY,'uicontextmenu',hcmenu_text);
            uimenu('parent',hcmenu_text,'label','Rotate',...
                   'callback',@(o,e)hobj.uiSetYRot());
            %Parse Param-Value
            p = inputParser;
            p.addParameter('Position',[axisXLim(1) + 0.1*axisXWidth, axisYLim(1) + 0.1*axisYWidth]);
            p.addParameter('Border',hobj.Border);
            p.addParameter('XUnit',hobj.XUnit);
            p.addParameter('YUnit',hobj.YUnit);
            p.addParameter('XLen',0.1*axisXWidth);
            p.addParameter('YLen',0.1*axisYWidth);
            p.addParameter('hTextX_Pos',0.02*[axisXWidth, -axisYWidth]);
            p.addParameter('hTextY_Pos',0.02*[-axisXWidth, axisYWidth]);
            p.addParameter('hTextY_Rot',hobj.hTextY_Rot);
            if isempty(varargin) %scalebar() 
                p.parse();
            elseif ~ishandle(varargin{1}) %scalebar('Prop','Value')
				p.parse(varargin{:});
            else %scalebar(hAxes,'Prop','Value')
                p.parse(varargin{2:end});
            end
			%default settings
            for prop={'XLen','YLen','XUnit','YUnit','hTextY_Rot',...
                      'Position','Border','hTextX_Pos','hTextY_Pos'}
                hobj.(prop{1}) = p.Results.(prop{1});
            end
        end
        function delete(hobj)
            delete(hobj.hLineX);
            delete(hobj.hLineY);
            delete(hobj.hTextX);
            delete(hobj.hTextY);
        end
        function uiSetLen(hobj,type,varargin)
            propname = [type 'Len'];
            answer = inputdlg('Enter length (number)',[type ' Len'],1, {num2str(hobj.(propname))});
            if isempty(answer); return; end
            len = str2double(answer{1});
            hobj.(propname) = len;
        end
        function uiSetUnit(hobj,type,varargin)
            propname = [type 'Unit'];
            answer = inputdlg('Enter unit (string)',[type ' Unit'],1, {hobj.(propname)});
            if isempty(answer); return; end
            unit = answer{1};
            hobj.(propname) = unit;
        end
        function uiSetYRot(hobj,varargin)
            hobj.hTextY_Rot = hobj.hTextY_Rot - 90;
        end
	end
	methods
	% Setting properties
		function SetPosition(hobj, varargin)
            value = hobj.Position;
			XPos = value(1);
			YPos = value(2);
			set(hobj.hLineY(1), 'XData', XPos*[1 1]);
 			set(hobj.hLineY(2), 'XData', (XPos+hobj.XLen)*[1 1]);
			set(hobj.hLineX(1), 'YData', YPos*[1 1]);
			set(hobj.hLineX(2), 'YData', (YPos+hobj.YLen)*[1 1]);
			set(hobj.hLineY, 'YData', YPos+[0 hobj.YLen]);
			set(hobj.hLineX, 'XData', XPos+[0 hobj.XLen]);
			set(hobj.hTextX, 'Position', [hobj.hTextX_Pos+value, 0]);
			set(hobj.hTextY, 'Position', [hobj.hTextY_Pos+value, 0]);
		end
		function SethTextX_Pos(hobj, varargin)
            value = hobj.hTextX_Pos;
			set(hobj.hTextX, 'Position', [hobj.Position+value, 0]);
		end
		function SethTextY_Pos(hobj,  varargin)
            value = hobj.hTextY_Pos;
			set(hobj.hTextY, 'Position', [hobj.Position+value, 0]);
        end
        function SethTextY_Rot(hobj, varargin)
            value = hobj.hTextY_Rot;
            set(hobj.hTextY, 'Rotation', value);
        end
		function SetBorder(hobj,  varargin)
            value = hobj.Border;
			XTyp = value(1);
			YTyp = value(2);
			switch upper(XTyp)
				case 'L'
					set(hobj.hLineX(1), 'Visible', 'on');
					set(hobj.hLineX(2), 'Visible', 'off');
				case 'U'
					set(hobj.hLineX(1), 'Visible', 'off');
					set(hobj.hLineX(2), 'Visible', 'on');
			end
			switch upper(YTyp)
				case 'L'
					set(hobj.hLineY(1), 'Visible', 'on');
					set(hobj.hLineY(2), 'Visible', 'off');
				case 'R'
					set(hobj.hLineY(1), 'Visible', 'off');
					set(hobj.hLineY(2), 'Visible', 'on');
            end
		end
		function SetXLen(hobj, varargin)
            value = hobj.XLen;
			XPos = hobj.Position(1);
			set(hobj.hLineX, 'XData', XPos+[0 value]);
			set(hobj.hLineY(2), 'XData', XPos*[1 1]+value);
            hobj.SetXUnit();
		end
		function SetYLen(hobj,  varargin)
            value = hobj.YLen;
			YPos = hobj.Position(2);
			set(hobj.hLineY, 'YData', YPos+[0 value]);
			set(hobj.hLineX(2), 'YData', YPos*[1 1]+value);
            hobj.SetYUnit();
		end
		function SetXUnit(hobj,  varargin)
            value = hobj.XUnit;
			if ishandle(hobj.hTextX)
				set(hobj.hTextX, 'String', [num2str(hobj.XLen),' ',value]);
            end
		end
		function SetYUnit(hobj,  varargin)
            value = hobj.YUnit;
			if ishandle(hobj.hTextY)
				set(hobj.hTextY, 'String',  [num2str(hobj.YLen),' ',value]);
            end
        end
	end
	methods (Access = private)
	%GUI dynamic drag
		function FcnStartDrag(hobj,varargin)
            pt = get(gca,'CurrentPoint');
			Pos.xpoint = pt(1,1);
			Pos.ypoint = pt(1,2);
			Pos.Position = hobj.Position;
            PosTX = hobj.hTextX_Pos;
            PosTY = hobj.hTextY_Pos;
			saveWindowFcn.Motion = get(gcf,'WindowButtonMotionFcn');
			saveWindowFcn.Up = get(gcf,'WindowButtonUpFcn');
			set(gcf,'WindowButtonMotionFcn',@(o,e)hobj.FcnDraging(Pos, PosTX, PosTY));
			set(gcf,'WindowButtonUpFcn',@(o,e)hobj.FcnEndDrag(saveWindowFcn));
		end
		function FcnDraging(hobj,Pos,PosTX,PosTY)
			pt = get(gca,'CurrentPoint');
			xpoint = pt(1,1);
			ypoint = pt(1,2);
			if isequal(gco, hobj.hTextX) %Drag Label ->  Drag Label Only
				hobj.hTextX_Pos = PosTX + [xpoint, ypoint] - [Pos.xpoint, Pos.ypoint];
			elseif isequal(gco, hobj.hTextY); 
				hobj.hTextY_Pos = PosTY + [xpoint, ypoint] - [Pos.xpoint, Pos.ypoint];
			else %Drag Line -> Drag Line & Label
				hobj.Position = Pos.Position + [xpoint, ypoint] - [Pos.xpoint, Pos.ypoint];
            end
		end
		function FcnEndDrag(hobj,saveWindowFcn)
			set(gcf,'pointer','arrow');
			set(gcf,'WindowButtonMotionFcn',saveWindowFcn.Motion);
			set(gcf,'WindowButtonUpFcn',saveWindowFcn.Up);
		end
	end

end
